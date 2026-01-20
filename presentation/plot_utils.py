import re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

def _strip_foam_header(text: str) -> str:
    # Remove FoamFile {...} block if present
    return re.sub(r"FoamFile\s*\{.*?\}\s*", "", text, flags=re.DOTALL)

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _extract_list_block(text: str) -> str:
    """
    Extract the content inside the main '( ... )' of OpenFOAM list files like
    points and faces (after header).
    """
    # remove comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = _strip_foam_header(text).strip()

    i = text.find("(")
    if i < 0:
        raise ValueError("Could not find list '(' in file.")

    depth = 0
    start = None
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
            if depth == 1:
                start = j + 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[start:j]
    raise ValueError("Unbalanced parentheses in list file.")

def read_points(points_path: str) -> np.ndarray:
    text = _read_text(Path(points_path))
    body = _extract_list_block(text)
    # points like: (x y z)
    pts = re.findall(r"\(\s*([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*\)", body)
    if not pts:
        raise ValueError("No points found.")
    return np.array(pts, dtype=float)

def read_faces(faces_path: str) -> list[list[int]]:
    text = _read_text(Path(faces_path))
    body = _extract_list_block(text)

    # faces lines like: 4(0 1 2 3) or 3(10 11 12)
    faces = []
    for n, inside in re.findall(r"(\d+)\s*\(\s*([0-9\s]+?)\s*\)", body):
        idx = [int(x) for x in inside.split()]
        # n is redundant but we can sanity check
        if len(idx) != int(n):
            # tolerate odd formatting; still keep what we got
            pass
        faces.append(idx)

    if not faces:
        raise ValueError("No faces found.")
    return faces

def read_boundary(boundary_path: str) -> dict:
    """
    Parse constant/polyMesh/boundary and return:
      patches[name] = {"type": str, "nFaces": int, "startFace": int}
    """
    text = _read_text(Path(boundary_path))
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = _strip_foam_header(text).strip()

    # extract main ( ... ) block
    i = text.find("(")
    if i < 0:
        raise ValueError("Could not find boundary '(' block.")
    depth = 0
    start = None
    end = None
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
            if depth == 1:
                start = j + 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise ValueError("Unbalanced parentheses in boundary file.")
    body = text[start:end]

    # Parse patch blocks: name { ... }
    patches = {}
    for m in re.finditer(r"\b([A-Za-z_]\w*)\b\s*\{", body):
        name = m.group(1)
        s = m.end() - 1  # at '{'
        d = 0
        bs = None
        be = None
        for j in range(s, len(body)):
            if body[j] == "{":
                d += 1
                if d == 1:
                    bs = j + 1
            elif body[j] == "}":
                d -= 1
                if d == 0:
                    be = j
                    break
        if be is None:
            continue
        block = body[bs:be]

        def get_int(key):
            mm = re.search(rf"\b{key}\b\s+(\d+)\s*;", block)
            return int(mm.group(1)) if mm else None

        def get_str(key):
            mm = re.search(rf"\b{key}\b\s+([A-Za-z_]\w*)\s*;", block)
            return mm.group(1) if mm else None

        nFaces = get_int("nFaces")
        startFace = get_int("startFace")
        ptype = get_str("type")

        if nFaces is not None and startFace is not None:
            patches[name] = {"type": ptype, "nFaces": nFaces, "startFace": startFace}

    if not patches:
        raise ValueError("No patches parsed from boundary file.")
    return patches


# -------------------------
# Geometry utilities
# -------------------------

def polygon_area_vector(pts: np.ndarray) -> np.ndarray:
    """
    Newell's method: returns area * normal vector (magnitude = 2*area for some formulations,
    but this implementation yields area vector with magnitude = area*2? We'll compute area from norm/2.
    """
    # pts shape (N,3), closed not required
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1); z2 = np.roll(z, -1)
    ax = np.sum((y - y2) * (z + z2))
    ay = np.sum((z - z2) * (x + x2))
    az = np.sum((x - x2) * (y + y2))
    return 0.5 * np.array([ax, ay, az], dtype=float)

def face_normal_from_points(pts: np.ndarray) -> np.ndarray:
    av = polygon_area_vector(pts)
    nrm = np.linalg.norm(av)
    if nrm == 0:
        return np.array([0.0, 0.0, 0.0])
    return av / nrm

def face_area_and_perimeter(pts: np.ndarray) -> tuple[float, float]:
    av = polygon_area_vector(pts)
    area = np.linalg.norm(av)  # Newell above already includes 0.5, so this is area
    diffs = np.roll(pts, -1, axis=0) - pts
    perim = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return float(area), float(perim)

def build_patch_faces(faces: list[list[int]], patches: dict) -> dict[str, list[list[int]]]:
    out = {}
    for name, meta in patches.items():
        s = meta["startFace"]
        n = meta["nFaces"]
        out[name] = faces[s:s+n]
    return out

def feature_edges_from_faces(face_list: list[list[int]], points: np.ndarray, crease_angle_deg=10.0) -> set[tuple[int, int]]:
    """
    Keep edges that are:
    - used by only one boundary face, or
    - shared by two boundary faces that are not coplanar (crease)
    Drops coplanar seams.
    """
    cos_thresh = np.cos(np.deg2rad(crease_angle_deg))
    edge_to_normals = {}

    for f in face_list:
        pts = points[np.array(f, dtype=int)]
        n = face_normal_from_points(pts)
        for a, b in zip(f, f[1:] + f[:1]):
            i, j = (a, b) if a < b else (b, a)
            edge_to_normals.setdefault((i, j), []).append(n)

    keep = set()
    for e, normals in edge_to_normals.items():
        if len(normals) == 1:
            keep.add(e)
        else:
            is_crease = False
            for i in range(len(normals)):
                for j in range(i+1, len(normals)):
                    dot = float(np.dot(normals[i], normals[j]))
                    if abs(dot) < cos_thresh:
                        is_crease = True
                        break
                if is_crease:
                    break
            if is_crease:
                keep.add(e)
    return keep

def edges_from_faces(face_list: list[list[int]]) -> set[tuple[int, int]]:
    edges = set()
    for f in face_list:
        for a, b in zip(f, f[1:] + f[:1]):
            i, j = (a, b) if a < b else (b, a)
            edges.add((i, j))
    return edges


def _unique_patch_points(face_list):
    idx = sorted({i for f in face_list for i in f})
    return idx

def _patch_bbox(P, face_list):
    idx = _unique_patch_points(face_list)
    pts = P[np.array(idx, dtype=int)]
    return pts.min(axis=0), pts.max(axis=0), pts

def _edges_polyline_coords(P, edges):
    xs, ys, zs = [], [], []
    for i, j in sorted(edges):
        x0, y0, z0 = P[i]
        x1, y1, z1 = P[j]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]
    return xs, ys, zs

def plot_polymesh_case(
    case_dir: str,
    unit: str = "mm",
    crease_angle_deg: float | None = 10.0,

    # patches to color/highlight
    inlet_patch: str = "inletMain",
    outlet_patch: str = "outlet",
    jet_patch: str = "jetInlet",

    # label placement
    label_shift_x_factor_big: float = 0.10,   # fraction of big size (move right)
    label_shift_x_factor_jet: float = 0.20,   # fraction of jet size (move right)
    label_lift_factor: float = 0.10,          # lift along patch normal

    show_global_Lx: bool = True,
    show_axes_labels_at_jet: bool = True,
):
    from pathlib import Path

    case_dir = Path(case_dir)
    pm = case_dir / "constant" / "polyMesh"

    points = read_points(str(pm / "points"))
    faces = read_faces(str(pm / "faces"))
    patches = read_boundary(str(pm / "boundary"))
    patch_faces = build_patch_faces(faces, patches)

    # units
    if unit == "mm":
        scale = 1000.0
        u = "mm"
    elif unit == "m":
        scale = 1.0
        u = "m"
    else:
        raise ValueError("unit must be 'mm' or 'm'")

    P = points * scale

    # boundary faces (all patches)
    all_boundary_faces = []
    for flist in patch_faces.values():
        all_boundary_faces.extend(flist)

    # boundary edges (clean)
    if crease_angle_deg is None:
        boundary_edges = edges_from_faces(all_boundary_faces)
    else:
        boundary_edges = feature_edges_from_faces(all_boundary_faces, P, crease_angle_deg=float(crease_angle_deg))

    fig = go.Figure()

    # Base boundary wireframe
    xs, ys, zs = _edges_polyline_coords(P, boundary_edges)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name="boundary",
        line=dict(width=6),
        hoverinfo="skip",
    ))

    # Helper: add colored patch outline
    def add_patch_outline(patch_name, width=12, show_legend=True):
        if patch_name not in patch_faces:
            return
        pedges = edges_from_faces(patch_faces[patch_name])
        hx, hy, hz = _edges_polyline_coords(P, pedges)
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode="lines",
            name=patch_name,
            line=dict(width=width),
            hoverinfo="skip",
            showlegend=show_legend
        ))

    # Color the important ones (Plotly default colors will differentiate them)
    add_patch_outline(inlet_patch, width=14, show_legend=True)
    add_patch_outline(outlet_patch, width=14, show_legend=True)
    add_patch_outline(jet_patch, width=14, show_legend=True)

    # Global extents (for Lx in title only)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    L = maxs - mins

    # ---- Labels: inletMain and jetInlet in 2 lines + axis hint line ----
    def add_size_label(patch_name, axes_hint: str, shift_x_factor: float):
        if patch_name not in patch_faces:
            return None

        pmin, pmax, pts = _patch_bbox(P, patch_faces[patch_name])
        dims = pmax - pmin

        # Determine which two axes span the patch (by picking the two largest extents)
        # For inletMain: should be y,z ; for jetInlet: x,y
        order = np.argsort(dims)  # ascending
        a, b = order[1], order[2]  # two largest dimensions
        da, db = dims[a], dims[b]

        # label text format: name \n a*b unit \n (axes)
        axis_names = ["x", "y", "z"]
        # enforce consistent order in label: for inletMain prefer y*z, for jetInlet prefer x*y
        # but generally keep as the two largest extents:
        axes_str = f"({axis_names[a]}*{axis_names[b]})"
        text = f"{patch_name}<br>{da:.2f}*{db:.2f} {u}<br>{axes_hint or axes_str}"

        # place it: patch center + lift along normal + shift to +x
        f0 = patch_faces[patch_name][0]
        n = face_normal_from_points(P[np.array(f0, dtype=int)])
        center = pts.mean(axis=0)

        size_ref = max(da, db, 1e-9)
        if(patch_name=="jetInlet"):
            pos = center + n * (label_lift_factor * size_ref) + np.array([shift_x_factor * size_ref, 0.0, 0.0])
        else:
            pos = center + n * (label_lift_factor * size_ref) + np.array([shift_x_factor * size_ref, 0.0, -4.0])

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="text",
            text=[text],
            showlegend=False
        ))
        return {"dims": dims, "center": center, "normal": n, "a": a, "b": b, "da": da, "db": db}

    # Force the axis hints exactly as you want
    inlet_info = add_size_label(inlet_patch, axes_hint="(y*z)", shift_x_factor=label_shift_x_factor_big)
    jet_info   = add_size_label(jet_patch,   axes_hint="(x*y)", shift_x_factor=label_shift_x_factor_jet)

    # x/y letters next to jet opening (optional)
    if show_axes_labels_at_jet and jet_info is not None:
        center = jet_info["center"]
        n = jet_info["normal"]
        # small offset scale
        ref = max(jet_info["da"], jet_info["db"], 1e-9)
        lift = 0.04 * ref
        step = 0.18 * ref

        base = center + n * lift
        px = base + np.array([step, 0.0, 0.0])
        py = base + np.array([0.0, step, 0.0])

        fig.add_trace(go.Scatter3d(x=[px[0]], y=[px[1]], z=[px[2]], mode="text", text=["x"], showlegend=False))
        fig.add_trace(go.Scatter3d(x=[py[0]], y=[py[1]], z=[py[2]], mode="text", text=["y"], showlegend=False))

    # Title
    title = f"{case_dir.name}"
    if show_global_Lx:
        title += f" | Lx={L[0]:.2f} {u}"

    fig.update_layout(
        title=None,
        scene=dict(
            aspectmode="data",
            xaxis_title=f"x [{u}]",
            yaxis_title=f"y [{u}]",
            zaxis_title=f"z [{u}]",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )

    info = {
        "unit": u,
        "L_global": L,
        "patch_names": sorted(patch_faces.keys()),
    }
    return fig, info
