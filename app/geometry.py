"""已完成任务某匿名系列的几何完整性检查（仅读取成功写出的匿名单帧实例）。

数据来源严格限定为“已完成任务 output_dir 中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**：

1. 收集该匿名 SeriesInstanceUID 的全部成功输出，逐个读取
   （``stop_before_pixels=True``，只取标签不取像素）；
2. 多帧实例（NumberOfFrames 显式大于 1 或不可解析）不参与几何分析，仅计数；
3. 校验 FrameOfReferenceUID、ImageOrientationPatient（6 个有限浮点方向余弦）
   与 ImagePositionPatient（3 个有限浮点坐标），缺失/无效即收集为
   “几何标签无效”冲突（附匿名 SOPInstanceUID 与原因）；
4. 用行/列方向余弦的叉积求法线并单位化；全部实例的
   FrameOfReferenceUID 必须唯一、法线方向必须一致（点积容差），
   否则收集冲突；
5. 坐标系与方向一致时，把 ImagePositionPatient 投影到法线并升序排序，
   投影位置差小于容差视为重合，收集冲突；
6. 存在任一冲突即整体报错（一次性列出全部冲突明细），不输出残缺排序；
7. 无冲突时计算相邻投影间距，以中位数为基准标记大于 1.5 倍中位数的间隙；
   单帧实例不足 3 张时无法以中位数判断间隙，在响应中注明。
"""
from __future__ import annotations

import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import pydicom

from .anonymizer import FileReport, JobResult

# 法线一致性：单位法线点积须 >= 1 - 该容差（约 2.6 度）
ORIENTATION_DOT_TOLERANCE = 1e-3
# 投影位置差（mm）小于该值视为重合
POSITION_COINCIDENCE_TOLERANCE_MM = 1e-3
# 间隙阈值：相邻投影间距 > 中位数 * 该倍数
GAP_RATIO_THRESHOLD = 1.5
# 至少需要这么多单帧实例才能用相邻间距中位数判断间隙
MIN_INSTANCES_FOR_GAP_ANALYSIS = 3

# 冲突类型（写入 detail.conflicts[].code）
CONFLICT_INVALID_TAGS = "INVALID_GEOMETRY_TAGS"
CONFLICT_FRAME_MISMATCH = "FRAME_OF_REFERENCE_MISMATCH"
CONFLICT_ORIENTATION = "ORIENTATION_INCONSISTENT"
CONFLICT_DUPLICATE_POSITION = "DUPLICATE_POSITION"

# 几何标签无效原因（写入 conflicts[].invalid_instances[].reasons）
REASON_MISSING_FOR_UID = "missing_frame_of_reference_uid"
REASON_MISSING_IOP = "missing_image_orientation_patient"
REASON_INVALID_IOP = "invalid_image_orientation_patient"
REASON_MISSING_IPP = "missing_image_position_patient"
REASON_INVALID_IPP = "invalid_image_position_patient"
REASON_DEGENERATE_ORIENTATION = "degenerate_orientation"

# 响应中浮点值的保留小数位（mm / 无量纲余弦，6 位足够且不丢有效信息）
_ROUND_DIGITS = 6


class GeometryError(Exception):
    """几何检查失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        series_uid: Optional[str] = None,
        conflicts: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        detail: dict = {"code": code, "message": message}
        if series_uid is not None:
            detail["series_instance_uid"] = series_uid
        if conflicts:
            detail["conflicts"] = conflicts
        self.detail = detail


@dataclass
class InstanceGeometry:
    """单帧实例的几何信息（法线已单位化）。"""

    sop_instance_uid: str
    output_file: str
    frame_of_reference_uid: str
    row_direction: tuple[float, float, float]
    column_direction: tuple[float, float, float]
    normal: tuple[float, float, float]
    position: tuple[float, float, float]


# --------------------------------------------------------------------------- #
# 向量运算
# --------------------------------------------------------------------------- #
def _cross(a: tuple, b: tuple) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: tuple) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _mean_normal(normals: list[tuple]) -> tuple[float, float, float]:
    """平均并重新单位化；一致性检查保证各法线几乎同向，平均不会退化。"""
    sx = sum(n[0] for n in normals)
    sy = sum(n[1] for n in normals)
    sz = sum(n[2] for n in normals)
    length = math.sqrt(sx * sx + sy * sy + sz * sz)
    if length < 1e-12:  # pragma: no cover - 一致性检查后不会出现
        return normals[0]
    return (sx / length, sy / length, sz / length)


def _round_vec(v: tuple) -> list[float]:
    return [round(c, _ROUND_DIGITS) for c in v]


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #
def _is_multi_value(value: object) -> bool:
    """pydicom 3.x 的 MultiValue 是 Sequence 而非 list 子类，统一按序列判断。"""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _parse_float_vector(value: object, count: int) -> Optional[tuple]:
    """把 DS 多值（MultiValue / 反斜杠字符串）解析为定长有限浮点元组。"""
    if value is None:
        return None
    if _is_multi_value(value):
        parts = list(value)
    else:
        parts = str(value).split("\\")
    if len(parts) != count:
        return None
    out: list[float] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return tuple(out)


def _vector_or_missing(ds, keyword: str, count: int) -> tuple[Optional[tuple], str]:
    """返回 (向量 | None, 状态)；状态为 ok | missing | invalid。"""
    value = getattr(ds, keyword, None)
    if value is None:
        return None, "missing"
    if _is_multi_value(value):
        if not list(value):
            return None, "missing"
    elif not str(value).strip():
        return None, "missing"
    vector = _parse_float_vector(value, count)
    if vector is None:
        return None, "invalid"
    return vector, "ok"


def _is_single_frame(ds) -> bool:
    """NumberOfFrames 缺省/为 1 视为单帧；显式大于 1 或不可解析一律排除。"""
    raw = getattr(ds, "NumberOfFrames", None)
    if raw is None or not str(raw).strip():
        return True
    try:
        return int(str(raw).strip()) == 1
    except (TypeError, ValueError):
        return False


def _extract_geometry(report: FileReport, ds) -> tuple[Optional[InstanceGeometry], Optional[dict]]:
    """从匿名数据集提取几何；返回 (geometry | None, invalid_problem | None)。"""
    reasons: list[str] = []
    messages: list[str] = []

    frame_uid = str(getattr(ds, "FrameOfReferenceUID", "") or "").strip()
    if not frame_uid:
        reasons.append(REASON_MISSING_FOR_UID)
        messages.append("缺少 FrameOfReferenceUID")

    iop, iop_status = _vector_or_missing(ds, "ImageOrientationPatient", 6)
    if iop_status == "missing":
        reasons.append(REASON_MISSING_IOP)
        messages.append("缺少 ImageOrientationPatient")
    elif iop_status == "invalid":
        reasons.append(REASON_INVALID_IOP)
        messages.append("ImageOrientationPatient 不是 6 个有限浮点方向余弦")

    ipp, ipp_status = _vector_or_missing(ds, "ImagePositionPatient", 3)
    if ipp_status == "missing":
        reasons.append(REASON_MISSING_IPP)
        messages.append("缺少 ImagePositionPatient")
    elif ipp_status == "invalid":
        reasons.append(REASON_INVALID_IPP)
        messages.append("ImagePositionPatient 不是 3 个有限浮点坐标")

    normal: Optional[tuple[float, float, float]] = None
    if iop is not None:
        row = (iop[0], iop[1], iop[2])
        col = (iop[3], iop[4], iop[5])
        cross = _cross(row, col)
        length = _length(cross)
        if length < 1e-12:
            reasons.append(REASON_DEGENERATE_ORIENTATION)
            messages.append("ImageOrientationPatient 行/列方向余弦平行，叉积法线退化")
        else:
            normal = (cross[0] / length, cross[1] / length, cross[2] / length)

    if reasons:
        problem = {
            "sop_instance_uid": report.sop_instance_uid,
            "output_file": report.output_name,
            "reasons": reasons,
            "message": "；".join(messages),
        }
        return None, problem

    return (
        InstanceGeometry(
            sop_instance_uid=report.sop_instance_uid,
            output_file=report.output_name,
            frame_of_reference_uid=frame_uid,
            row_direction=(iop[0], iop[1], iop[2]),
            column_direction=(iop[3], iop[4], iop[5]),
            normal=normal,
            position=(ipp[0], ipp[1], ipp[2]),
        ),
        None,
    )


# --------------------------------------------------------------------------- #
# 输出收集
# --------------------------------------------------------------------------- #
def _successful_series_reports(result: JobResult, series_uid: str) -> list[FileReport]:
    """该匿名系列内真正写出匿名文件、且记录了匿名 SOP UID 的报告。"""
    return [
        report
        for report in result.files
        if report.status != "error"
        and report.output_name
        and report.sop_instance_uid
        and report.series_instance_uid == series_uid
    ]


def _find_coincident_groups(
    projections: list[tuple[float, InstanceGeometry]],
) -> list[list[tuple[float, InstanceGeometry]]]:
    """在按投影升序的列表上找出位置重合（相邻差 < 容差）的连续组。"""
    groups: list[list[tuple[float, InstanceGeometry]]] = []
    current = [projections[0]]
    for item in projections[1:]:
        if item[0] - current[-1][0] < POSITION_COINCIDENCE_TOLERANCE_MM:
            current.append(item)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [item]
    if len(current) > 1:
        groups.append(current)
    return groups


def _sorted_projections(
    geometries: list[InstanceGeometry], reference_normal: tuple
) -> list[tuple[float, InstanceGeometry]]:
    """把 ImagePositionPatient 投影到法线并按投影升序（SOP UID 决胜保证确定）。"""
    return sorted(
        ((_dot(g.position, reference_normal), g) for g in geometries),
        key=lambda item: (item[0], item[1].sop_instance_uid),
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def analyze_series_geometry(job, series_uid: str) -> dict:
    """分析某匿名系列单帧实例的几何完整性；任何冲突/不可用均抛 GeometryError。"""
    result: JobResult = job.result
    reports = _successful_series_reports(result, series_uid)
    if not reports:
        raise GeometryError(
            "SERIES_NOT_FOUND",
            404,
            "该匿名 SeriesInstanceUID 在本任务的成功输出中不存在",
            series_uid=series_uid,
        )

    geometries: list[InstanceGeometry] = []
    invalid_instances: list[dict] = []
    multiframe_count = 0
    single_frame_count = 0

    for report in sorted(reports, key=lambda r: r.sop_instance_uid):
        path = os.path.join(result.output_dir, report.output_name)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as exc:  # 自己写出的匿名文件不可读/已丢失属于服务端异常
            raise GeometryError(
                "ANONYMIZED_OUTPUT_UNREADABLE",
                500,
                f"无法读取匿名输出文件 {report.output_name}：{exc}",
                series_uid=series_uid,
            ) from exc
        if not _is_single_frame(ds):
            multiframe_count += 1
            continue
        single_frame_count += 1
        geometry, problem = _extract_geometry(report, ds)
        if problem is not None:
            invalid_instances.append(problem)
        else:
            geometries.append(geometry)

    if single_frame_count == 0:
        raise GeometryError(
            "NO_SINGLE_FRAME_INSTANCES",
            409,
            f"该系列的成功输出均为多帧实例（{multiframe_count} 个），"
            "几何检查仅支持单帧实例",
            series_uid=series_uid,
        )

    conflicts: list[dict] = []
    if invalid_instances:
        conflicts.append(
            {
                "code": CONFLICT_INVALID_TAGS,
                "message": f"{len(invalid_instances)} 个单帧实例的几何标签缺失或无效",
                "sop_instance_uids": sorted(
                    {p["sop_instance_uid"] for p in invalid_instances}
                ),
                "invalid_instances": invalid_instances,
                "frame_of_reference_groups": [],
                "orientations": [],
                "duplicate_positions": [],
            }
        )

    # FrameOfReferenceUID 一致性（坐标系不同则投影位置不可比）
    frame_groups: dict[str, list[str]] = {}
    for g in geometries:
        frame_groups.setdefault(g.frame_of_reference_uid, []).append(
            g.sop_instance_uid
        )
    frame_mismatch = len(frame_groups) > 1
    if frame_mismatch:
        conflicts.append(
            {
                "code": CONFLICT_FRAME_MISMATCH,
                "message": (
                    f"单帧实例分属 {len(frame_groups)} 个不同的 "
                    "FrameOfReferenceUID，无法在同一坐标系内排序"
                ),
                "sop_instance_uids": sorted(g.sop_instance_uid for g in geometries),
                "invalid_instances": [],
                "frame_of_reference_groups": [
                    {
                        "frame_of_reference_uid": uid,
                        "sop_instance_uids": sorted(uids),
                    }
                    for uid, uids in sorted(frame_groups.items())
                ],
                "orientations": [],
                "duplicate_positions": [],
            }
        )

    # 方向一致性（单位法线点积容差；反向法线同样视为不一致）
    orientation_consistent = True
    if geometries:
        reference = geometries[0].normal
        orientation_consistent = all(
            _dot(reference, g.normal) >= 1.0 - ORIENTATION_DOT_TOLERANCE
            for g in geometries
        )
    if not orientation_consistent:
        conflicts.append(
            {
                "code": CONFLICT_ORIENTATION,
                "message": (
                    "单帧实例的 ImageOrientationPatient 方向不一致"
                    "（法线夹角超过容差），无法统一排序"
                ),
                "sop_instance_uids": sorted(g.sop_instance_uid for g in geometries),
                "invalid_instances": [],
                "frame_of_reference_groups": [],
                "orientations": [
                    {
                        "sop_instance_uid": g.sop_instance_uid,
                        "row_direction": _round_vec(g.row_direction),
                        "column_direction": _round_vec(g.column_direction),
                        "normal": _round_vec(g.normal),
                    }
                    for g in geometries
                ],
                "duplicate_positions": [],
            }
        )

    # 投影位置重合（仅在坐标系唯一且方向一致时，投影才有可比性）
    if geometries and not frame_mismatch and orientation_consistent:
        reference_normal = _mean_normal([g.normal for g in geometries])
        projections = _sorted_projections(geometries, reference_normal)
        coincident = _find_coincident_groups(projections)
        if coincident:
            conflicts.append(
                {
                    "code": CONFLICT_DUPLICATE_POSITION,
                    "message": (
                        f"发现 {len(coincident)} 组投影位置重合的单帧实例"
                        f"（相邻投影差 < {POSITION_COINCIDENCE_TOLERANCE_MM} mm）"
                    ),
                    "sop_instance_uids": sorted(
                        g.sop_instance_uid
                        for group in coincident
                        for _, g in group
                    ),
                    "invalid_instances": [],
                    "frame_of_reference_groups": [],
                    "orientations": [],
                    "duplicate_positions": [
                        {
                            "projection": round(group[0][0], _ROUND_DIGITS),
                            "sop_instance_uids": sorted(
                                g.sop_instance_uid for _, g in group
                            ),
                        }
                        for group in coincident
                    ],
                }
            )

    if conflicts:
        raise GeometryError(
            "GEOMETRY_CONFLICT",
            409,
            "系列单帧实例的几何存在冲突，无法给出一致的切片排序与间隙分析",
            series_uid=series_uid,
            conflicts=conflicts,
        )

    # ---- 无冲突：排序、相邻间距、以中位数为基准的间隙标记 ----------------- #
    reference_normal = _mean_normal([g.normal for g in geometries])
    projections = _sorted_projections(geometries, reference_normal)

    instances: list[dict] = []
    spacings: list[float] = []
    for index, (projection, g) in enumerate(projections):
        spacing_to_next: Optional[float] = None
        if index + 1 < len(projections):
            spacing = projections[index + 1][0] - projection
            spacings.append(spacing)
            spacing_to_next = round(spacing, _ROUND_DIGITS)
        instances.append(
            {
                "index": index,
                "sop_instance_uid": g.sop_instance_uid,
                "projection": round(projection, _ROUND_DIGITS),
                "spacing_to_next": spacing_to_next,
            }
        )

    instance_count = len(instances)
    gap_analysis_possible = instance_count >= MIN_INSTANCES_FOR_GAP_ANALYSIS
    median_spacing: Optional[float] = None
    gap_threshold: Optional[float] = None
    anomaly_indices: list[int] = []
    gaps: list[dict] = []
    gap_note: Optional[str] = None

    if gap_analysis_possible:
        # 重合位置已在冲突阶段拒绝，此处中位数必然为正
        median = statistics.median(spacings)
        threshold = median * GAP_RATIO_THRESHOLD
        median_spacing = round(median, _ROUND_DIGITS)
        gap_threshold = round(threshold, _ROUND_DIGITS)
        for index, spacing in enumerate(spacings):
            if spacing > threshold:
                anomaly_indices.append(index)
                gaps.append(
                    {
                        "index": index,
                        "spacing": round(spacing, _ROUND_DIGITS),
                        "ratio": round(spacing / median, _ROUND_DIGITS),
                        "before_sop_instance_uid": instances[index][
                            "sop_instance_uid"
                        ],
                        "after_sop_instance_uid": instances[index + 1][
                            "sop_instance_uid"
                        ],
                    }
                )
    else:
        gap_note = (
            f"系列内单帧实例不足 {MIN_INSTANCES_FOR_GAP_ANALYSIS} 张"
            f"（当前 {instance_count} 张），无法以相邻间距中位数判断间隙"
        )

    return {
        "job_id": job.id,
        "series_instance_uid": series_uid,
        "study_instance_uid": reports[0].study_instance_uid,
        "frame_of_reference_uid": geometries[0].frame_of_reference_uid,
        "instance_count": instance_count,
        "skipped_multiframe_count": multiframe_count,
        "row_direction": _round_vec(geometries[0].row_direction),
        "column_direction": _round_vec(geometries[0].column_direction),
        "normal": _round_vec(reference_normal),
        "median_spacing": median_spacing,
        "gap_threshold": gap_threshold,
        "gap_analysis_possible": gap_analysis_possible,
        "gap_note": gap_note,
        "instances": instances,
        "anomaly_indices": anomaly_indices,
        "gaps": gaps,
    }
