"""系列几何完整性检查：法线排序、间隙标记、四类冲突与 404/409 错误路径。"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from app.crypto import derive_uid
from tests.sample_data import _write_bytes

SECRET = "geometry-secret-123456"
STUDY_UID = "3.1"
SERIES_UID = "2.1"
FRAME_UID = "4.1"
AXIAL_IOP = "1\\0\\0\\0\\1\\0"  # 行 (1,0,0)，列 (0,1,0) -> 法线 (0,0,1)


def _image_bytes(
    sop_uid: str,
    *,
    iop=AXIAL_IOP,
    ipp: str = "0\\0\\0",
    frame_uid: str | None = FRAME_UID,
    series_uid: str = SERIES_UID,
    study_uid: str = STUDY_UID,
    number_of_frames=None,
    raw_ipp: bytes | None = None,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = "CT"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    if frame_uid is not None:
        ds.FrameOfReferenceUID = frame_uid
    if iop is not None:
        ds.ImageOrientationPatient = iop
    if ipp is not None:
        ds.ImagePositionPatient = ipp
    if raw_ipp is not None:
        # pydicom 3.x 会拒绝构造非法 DS 元素，改用 RawDataElement 注入坏字节
        from pydicom.dataelem import RawDataElement
        from pydicom.tag import Tag

        ds[Tag(0x00200032)] = RawDataElement(
            Tag(0x00200032), "DS", len(raw_ipp), raw_ipp, 0, True, None
        )
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    return _write_bytes(ds)


def _axial_series(sop_uids: list[str], positions: list[float], **kwargs) -> dict:
    """轴位系列：IPP = (0, 0, z)，上传顺序刻意打乱。"""
    files = {}
    order = np.random.RandomState(42).permutation(len(sop_uids))
    for i in order:
        files[f"slice_{sop_uids[i].split('.')[-1]}.dcm"] = _image_bytes(
            sop_uids[i], ipp=f"0\\0\\{positions[i]}", **kwargs
        )
    return files


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload(client, files: dict[str, bytes]):
    r = client.post(
        "/jobs",
        files={"file": ("studies.zip", _zip_bytes(files), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _geometry(client, job_id: str, series_uid: str):
    return client.get(f"/jobs/{job_id}/series/{series_uid}/geometry")


def _anon(original: str) -> str:
    return derive_uid(SECRET, original)


ANON_SERIES = derive_uid(SECRET, SERIES_UID)
ANON_FRAME = derive_uid(SECRET, FRAME_UID)


# --------------------------------------------------------------------------- #
# 成功路径：排序 / 间距 / 间隙
# --------------------------------------------------------------------------- #
def test_geometry_sorted_uniform_spacing(client):
    sops = ["1.1.1", "1.1.2", "1.1.3", "1.1.4"]
    files = _axial_series(sops, [0.0, 5.0, 10.0, 15.0])
    job = _upload(client, files)

    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job["id"]
    assert body["series_instance_uid"] == ANON_SERIES
    assert body["study_instance_uid"] == _anon(STUDY_UID)
    assert body["frame_of_reference_uid"] == ANON_FRAME
    assert body["instance_count"] == 4
    assert body["skipped_multiframe_count"] == 0
    assert body["row_direction"] == [1.0, 0.0, 0.0]
    assert body["column_direction"] == [0.0, 1.0, 0.0]
    assert body["normal"] == [0.0, 0.0, 1.0]

    # 按投影升序：匿名 SOP UID 顺序对应 z = 0, 5, 10, 15
    assert [i["sop_instance_uid"] for i in body["instances"]] == [
        _anon(s) for s in sops
    ]
    assert [i["projection"] for i in body["instances"]] == [0.0, 5.0, 10.0, 15.0]
    assert [i["index"] for i in body["instances"]] == [0, 1, 2, 3]
    assert [i["spacing_to_next"] for i in body["instances"]] == [5.0, 5.0, 5.0, None]

    assert body["gap_analysis_possible"] is True
    assert body["gap_note"] is None
    assert body["median_spacing"] == 5.0
    assert body["gap_threshold"] == 7.5
    assert body["anomaly_indices"] == []
    assert body["gaps"] == []


def test_geometry_gap_detected(client):
    sops = ["1.2.1", "1.2.2", "1.2.3", "1.2.4"]
    files = _axial_series(sops, [0.0, 5.0, 10.0, 25.0])  # 间距 5, 5, 15
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["median_spacing"] == 5.0
    assert body["gap_threshold"] == 7.5
    assert body["anomaly_indices"] == [2]
    assert len(body["gaps"]) == 1
    gap = body["gaps"][0]
    assert gap["index"] == 2
    assert gap["spacing"] == 15.0
    assert gap["ratio"] == 3.0
    assert gap["before_sop_instance_uid"] == _anon("1.2.3")
    assert gap["after_sop_instance_uid"] == _anon("1.2.4")


def test_geometry_exactly_threshold_not_flagged(client):
    # 间距 5, 5, 7.5：中位数 5，阈值 7.5；等于阈值不算异常（严格大于）
    sops = ["1.3.1", "1.3.2", "1.3.3", "1.3.4"]
    files = _axial_series(sops, [0.0, 5.0, 10.0, 17.5])
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["gap_threshold"] == 7.5
    assert body["anomaly_indices"] == []
    assert body["gaps"] == []


def test_geometry_two_instances_cannot_judge_gaps(client):
    sops = ["1.4.1", "1.4.2"]
    files = _axial_series(sops, [0.0, 100.0])  # 巨大间距也无法判断
    job = _upload(client, files)

    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["instance_count"] == 2
    assert body["gap_analysis_possible"] is False
    assert "不足 3 张" in body["gap_note"]
    assert body["median_spacing"] is None
    assert body["gap_threshold"] is None
    assert body["anomaly_indices"] == []
    assert body["gaps"] == []
    # 排序与相邻间距仍然给出
    assert [i["sop_instance_uid"] for i in body["instances"]] == [
        _anon("1.4.1"),
        _anon("1.4.2"),
    ]
    assert body["instances"][0]["spacing_to_next"] == 100.0
    assert body["instances"][1]["spacing_to_next"] is None


def test_geometry_single_instance(client):
    files = {"one.dcm": _image_bytes("1.5.1", ipp="3\\4\\5")}
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["instance_count"] == 1
    assert body["gap_analysis_possible"] is False
    assert "不足 3 张" in body["gap_note"]
    assert body["instances"][0]["projection"] == 5.0
    assert body["instances"][0]["spacing_to_next"] is None


def test_geometry_oblique_orientation(client):
    # 斜位：行 (1,0,0)，列 (0,0.6,0.8) -> 法线 (0,-0.8,0.6)
    iop = "1\\0\\0\\0\\0.6\\0.8"
    sops = ["1.6.1", "1.6.2", "1.6.3"]
    # 沿法线方向摆放：pos = t * (0,-0.8,0.6)，t = 0, 5, 10
    files = {}
    for sop, t in zip(sops, [0.0, 5.0, 10.0]):
        files[f"{sop}.dcm"] = _image_bytes(
            sop, iop=iop, ipp=f"0\\{-0.8 * t}\\{0.6 * t}"
        )
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["normal"] == [0.0, -0.8, 0.6]
    projections = [i["projection"] for i in body["instances"]]
    assert projections == pytest.approx([0.0, 5.0, 10.0], abs=1e-6)
    assert [i["sop_instance_uid"] for i in body["instances"]] == [
        _anon(s) for s in sops
    ]
    assert body["median_spacing"] == 5.0


def test_geometry_negative_normal_sorts_along_normal(client):
    # 列翻转 -> 法线 (0,0,-1)：排序沿法线方向（投影 = -z），与 z 轴数值方向相反
    iop = "1\\0\\0\\0\\-1\\0"
    sops = ["1.7.1", "1.7.2", "1.7.3"]
    files = {}
    for sop, z in zip(sops, [-10.0, -5.0, 0.0]):
        files[f"{sop}.dcm"] = _image_bytes(sop, iop=iop, ipp=f"0\\0\\{z}")
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["normal"] == [0.0, 0.0, -1.0]
    # 投影 = dot((0,0,z), (0,0,-1)) = -z -> 升序为 z=0, -5, -10
    assert [i["sop_instance_uid"] for i in body["instances"]] == [
        _anon("1.7.3"),
        _anon("1.7.2"),
        _anon("1.7.1"),
    ]
    assert [i["projection"] for i in body["instances"]] == [0.0, 5.0, 10.0]


def test_geometry_inplane_offset_does_not_affect_projection(client):
    # 面内偏移（x/y 不同）不影响法线投影
    sops = ["1.8.1", "1.8.2", "1.8.3"]
    files = {
        "a.dcm": _image_bytes(sops[0], ipp="10\\20\\0"),
        "b.dcm": _image_bytes(sops[1], ipp="-5\\3\\5"),
        "c.dcm": _image_bytes(sops[2], ipp="7\\8\\10"),
    }
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert [i["projection"] for i in body["instances"]] == [0.0, 5.0, 10.0]
    assert [i["sop_instance_uid"] for i in body["instances"]] == [
        _anon(s) for s in sops
    ]


def test_geometry_tiny_orientation_deviation_within_tolerance(client):
    # 约 0.057 度的微小方向偏差仍在容差内，不判冲突
    sops = ["1.9.1", "1.9.2", "1.9.3"]
    files = {
        "a.dcm": _image_bytes(sops[0], ipp="0\\0\\0"),
        "b.dcm": _image_bytes(sops[1], ipp="0\\0\\5"),
        "c.dcm": _image_bytes(
            sops[2], iop="1\\0\\0\\0\\0.9999995\\0.001", ipp="0\\0\\10"
        ),
    }
    job = _upload(client, files)

    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 200, r.text
    assert r.json()["instance_count"] == 3


def test_geometry_multiframe_skipped(client):
    sops = ["1.10.1", "1.10.2", "1.10.3"]
    files = _axial_series(sops, [0.0, 5.0, 10.0])
    files["mf.dcm"] = _image_bytes("1.10.9", ipp="0\\0\\7.5", number_of_frames=2)
    job = _upload(client, files)

    body = _geometry(client, job["id"], ANON_SERIES).json()
    assert body["instance_count"] == 3
    assert body["skipped_multiframe_count"] == 1
    assert [i["projection"] for i in body["instances"]] == [0.0, 5.0, 10.0]


# --------------------------------------------------------------------------- #
# 409：几何冲突
# --------------------------------------------------------------------------- #
def _expect_conflict(client, job, code: str):
    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "GEOMETRY_CONFLICT"
    assert detail["series_instance_uid"] == ANON_SERIES
    conflicts = {c["code"]: c for c in detail["conflicts"]}
    assert code in conflicts
    return conflicts[code]


def test_geometry_orientation_conflict(client):
    files = {
        "a.dcm": _image_bytes("1.11.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.11.2", ipp="0\\0\\5"),
        # 冠状位方向：法线 (0,-1,0)，与轴位不一致
        "c.dcm": _image_bytes("1.11.3", iop="1\\0\\0\\0\\0\\1", ipp="0\\0\\10"),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "ORIENTATION_INCONSISTENT")
    assert sorted(conflict["sop_instance_uids"]) == sorted(
        _anon(s) for s in ("1.11.1", "1.11.2", "1.11.3")
    )
    normals = {tuple(o["normal"]) for o in conflict["orientations"]}
    assert normals == {(0.0, 0.0, 1.0), (0.0, -1.0, 0.0)}


def test_geometry_antiparallel_orientation_conflict(client):
    # 法线 (0,0,-1) 与 (0,0,1) 反向同样视为方向不一致
    files = {
        "a.dcm": _image_bytes("1.12.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.12.2", ipp="0\\0\\5"),
        "c.dcm": _image_bytes("1.12.3", iop="1\\0\\0\\0\\-1\\0", ipp="0\\0\\10"),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "ORIENTATION_INCONSISTENT")
    normals = {tuple(o["normal"]) for o in conflict["orientations"]}
    assert normals == {(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)}


def test_geometry_frame_of_reference_mismatch(client):
    files = {
        "a.dcm": _image_bytes("1.13.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.13.2", ipp="0\\0\\5"),
        "c.dcm": _image_bytes("1.13.3", ipp="0\\0\\10", frame_uid="4.2"),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "FRAME_OF_REFERENCE_MISMATCH")
    groups = conflict["frame_of_reference_groups"]
    assert len(groups) == 2
    by_frame = {g["frame_of_reference_uid"]: g["sop_instance_uids"] for g in groups}
    assert by_frame[ANON_FRAME] == sorted([_anon("1.13.1"), _anon("1.13.2")])
    assert by_frame[_anon("4.2")] == [_anon("1.13.3")]


def test_geometry_invalid_geometry_tags(client):
    files = {
        "ok1.dcm": _image_bytes("1.14.1", ipp="0\\0\\0"),
        "ok2.dcm": _image_bytes("1.14.2", ipp="0\\0\\5"),
        "no_iop.dcm": _image_bytes("1.14.3", iop=None, ipp="0\\0\\10"),
        "bad_ipp.dcm": _image_bytes("1.14.4", ipp=None, raw_ipp=b"abc\\0\\1"),
        "no_frame.dcm": _image_bytes("1.14.5", ipp="0\\0\\15", frame_uid=None),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "INVALID_GEOMETRY_TAGS")
    by_sop = {i["sop_instance_uid"]: i for i in conflict["invalid_instances"]}
    assert set(by_sop) == {_anon("1.14.3"), _anon("1.14.4"), _anon("1.14.5")}
    assert by_sop[_anon("1.14.3")]["reasons"] == [
        "missing_image_orientation_patient"
    ]
    assert by_sop[_anon("1.14.4")]["reasons"] == ["invalid_image_position_patient"]
    assert by_sop[_anon("1.14.5")]["reasons"] == ["missing_frame_of_reference_uid"]
    assert all(i["output_file"].endswith(".dcm") for i in by_sop.values())


def test_geometry_degenerate_orientation(client):
    # 行/列方向余弦平行 -> 叉积为零向量
    files = {
        "a.dcm": _image_bytes("1.15.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.15.2", ipp="0\\0\\5"),
        "c.dcm": _image_bytes("1.15.3", iop="1\\0\\0\\1\\0\\0", ipp="0\\0\\10"),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "INVALID_GEOMETRY_TAGS")
    reasons = conflict["invalid_instances"][0]["reasons"]
    assert reasons == ["degenerate_orientation"]
    assert conflict["invalid_instances"][0]["sop_instance_uid"] == _anon("1.15.3")


def test_geometry_duplicate_position(client):
    files = {
        "a.dcm": _image_bytes("1.16.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.16.2", ipp="0\\0\\0"),  # 与 a 重合
        "c.dcm": _image_bytes("1.16.3", ipp="0\\0\\5"),
    }
    job = _upload(client, files)

    conflict = _expect_conflict(client, job, "DUPLICATE_POSITION")
    assert conflict["duplicate_positions"] == [
        {
            "projection": 0.0,
            "sop_instance_uids": sorted([_anon("1.16.1"), _anon("1.16.2")]),
        }
    ]
    assert conflict["sop_instance_uids"] == sorted(
        [_anon("1.16.1"), _anon("1.16.2")]
    )


def test_geometry_multiple_conflicts_reported_together(client):
    # 无效标签 + 坐标系不一致一次性全部列出
    files = {
        "a.dcm": _image_bytes("1.17.1", ipp="0\\0\\0"),
        "b.dcm": _image_bytes("1.17.2", ipp="0\\0\\5", frame_uid="4.2"),
        "c.dcm": _image_bytes("1.17.3", iop=None, ipp="0\\0\\10"),
    }
    job = _upload(client, files)

    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 409, r.text
    codes = {c["code"] for c in r.json()["detail"]["conflicts"]}
    assert codes == {"INVALID_GEOMETRY_TAGS", "FRAME_OF_REFERENCE_MISMATCH"}


def test_geometry_all_multiframe_409(client):
    files = {
        "mf1.dcm": _image_bytes("1.18.1", ipp="0\\0\\0", number_of_frames=2),
        "mf2.dcm": _image_bytes("1.18.2", ipp="0\\0\\5", number_of_frames=3),
    }
    job = _upload(client, files)

    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "NO_SINGLE_FRAME_INSTANCES"
    assert detail["series_instance_uid"] == ANON_SERIES


# --------------------------------------------------------------------------- #
# 404 / 409：任务与系列
# --------------------------------------------------------------------------- #
def test_geometry_job_not_found_404(client):
    r = _geometry(client, "deadbeef", ANON_SERIES)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "JOB_NOT_FOUND"
    assert detail["series_instance_uid"] == ANON_SERIES


def test_geometry_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": SECRET},
    )
    job = r.json()
    assert job["status"] == "failed"
    resp = _geometry(client, job["id"], ANON_SERIES)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NOT_READY"


def test_geometry_series_not_found_404(client):
    files = {"a.dcm": _image_bytes("1.19.1", ipp="0\\0\\0")}
    job = _upload(client, files)
    r = _geometry(client, job["id"], "2.25.unknown-anon-series")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "SERIES_NOT_FOUND"
    assert detail["series_instance_uid"] == "2.25.unknown-anon-series"


def test_geometry_series_without_successful_output_404(client):
    # 系列内唯一文件是非法 DICOM -> 该系列无成功输出
    files = {
        "good.dcm": _image_bytes("1.20.1", ipp="0\\0\\0"),
        "broken/bad.dcm": b"\x00\x01\x02not dicom\xff\xfe",
    }
    job = _upload(client, files)
    assert job["error_count"] == 1
    # 存在的系列可正常响应；不存在的系列 404
    ok = _geometry(client, job["id"], ANON_SERIES)
    assert ok.status_code == 200
    missing = _geometry(client, job["id"], _anon("9.9.9"))
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "SERIES_NOT_FOUND"


def test_geometry_output_file_removed_500(client):
    files = {"a.dcm": _image_bytes("1.21.1", ipp="0\\0\\0")}
    job = _upload(client, files)
    import os

    from app.main import registry

    job_obj = registry.get(job["id"])
    for name in os.listdir(job_obj.result.output_dir):
        os.remove(os.path.join(job_obj.result.output_dir, name))
    r = _geometry(client, job["id"], ANON_SERIES)
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "ANONYMIZED_OUTPUT_UNREADABLE"


# --------------------------------------------------------------------------- #
# 集成：标准样例 ZIP（CT 单帧系列 + RTSTRUCT 系列）
# --------------------------------------------------------------------------- #
def test_geometry_with_canonical_sample_zip(client, tmp_path):
    from tests.sample_data import build_sample_zip

    zip_path = tmp_path / "sample.zip"
    meta = build_sample_zip(str(zip_path), seed=1)
    with open(zip_path, "rb") as fh:
        r = client.post(
            "/jobs",
            files={"file": ("study.zip", fh, "application/zip")},
            data={"secret": SECRET, "shift_dates": "false"},
        )
    assert r.status_code == 201, r.text
    job = r.json()

    # 样例 CT 未写几何标签 -> 两个单帧实例均 INVALID_GEOMETRY_TAGS
    r = _geometry(client, job["id"], _anon(meta["series_uid"]))
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "GEOMETRY_CONFLICT"
    conflict = detail["conflicts"][0]
    assert conflict["code"] == "INVALID_GEOMETRY_TAGS"
    assert sorted(conflict["sop_instance_uids"]) == sorted(
        _anon(s) for s in meta["sops"]
    )
    reasons = {
        reason
        for inst in conflict["invalid_instances"]
        for reason in inst["reasons"]
    }
    assert "missing_image_orientation_patient" in reasons
    assert "missing_image_position_patient" in reasons

    # RTSTRUCT 系列（单帧、无几何标签）同样报无效标签冲突
    rt = _geometry(client, job["id"], _anon("1.2.3.4.5.6.7.8.9.1.7"))
    assert rt.status_code == 409
    assert rt.json()["detail"]["conflicts"][0]["code"] == "INVALID_GEOMETRY_TAGS"
