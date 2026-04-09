# app_centroid_net.py
# Streamlit app for centroid prediction using DSNT-trained TinyUNetLogits
#
# Teaching focus:
#   (1) Network probability map (spatial softmax)
#   (2) DSNT centroid (expectation over probability map)
#   (3) ROI box centered at centroid with size = image_size / 2 (auto; scale-consistent)
#
# Inputs (FULL FOV only; NOT cropped):
#   - Cine CMR frames (2CH / 3CH / 4CH / SAX) as PNG/JPG/TIF/BMP
#   - MATLAB .mat containing variable `image` (HxW or HxWxT)
#
# Coordinate convention (matches training):
#   - 1-based coordinates: x = column in [1..W], y = row in [1..H]
#
# Run:
#   cd D:\Course Review 07192024\app_centroid_net
#   streamlit run app_centroid_net.py
#
# Requirements:
#   pip install streamlit torch opencv-python scipy numpy pillow
#
# Checkpoint handling (NO absolute paths required):
#   - By default, this app looks for:
#       best_centroid_xy_net_DSNT_EPE.pt
#     in the SAME folder as this script.
#   - You may still browse/select a checkpoint, but relative is the default.

import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from collections import OrderedDict
from PIL import ImageDraw

# ============================================================
# Model definitions (MUST match training)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class FeatureHook:
    """Capture intermediate activations by module name."""
    def __init__(self, model: nn.Module, layer_names):
        self.model = model
        self.layer_names = set(layer_names)
        self.handles = []
        self.feats = OrderedDict()

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            # out: torch.Tensor [B,C,H,W]
            self.feats[name] = out.detach()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        
class TinyUNetLogits(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bot = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)  # logits

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bot(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        z = self.out(d1)  # [B,1,H,W]
        return z

# ============================================================
# Pre/post utilities (match training)
# ============================================================

def mad(x: np.ndarray, eps: float = 1e-6) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + eps)

def robust_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img - np.median(img)
    img = img / (mad(img) + 1e-6)
    return img

def spatial_softmax_2d_logits(z: torch.Tensor, beta: float) -> torch.Tensor:
    """
    z: [B,1,H,W] logits -> p: [B,1,H,W] prob map (sum-to-1 per sample)
    p = exp(beta*(z-max))/sum(exp(beta*(z-max)))
    beta (β) controls sharpness: larger β => peakier distribution.
    """
    B, C, H, W = z.shape
    z2 = z.view(B, -1)
    z2 = z2 - z2.max(dim=1, keepdim=True).values
    p = torch.exp(beta * z2)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    return p.view(B, 1, H, W)

def expected_xy_from_prob(p: torch.Tensor) -> torch.Tensor:
    """
    p: [B,1,H,W], sum-to-1
    return: [B,2] in 1-based (x=col, y=row)
    """
    B, _, H, W = p.shape
    device = p.device
    xs = torch.arange(1, W + 1, device=device, dtype=p.dtype).view(1, 1, 1, W)
    ys = torch.arange(1, H + 1, device=device, dtype=p.dtype).view(1, 1, H, 1)
    x = (p * xs).sum(dim=(1, 2, 3))
    y = (p * ys).sum(dim=(1, 2, 3))
    return torch.stack([x, y], dim=1)

def hard_argmax_xy(p: torch.Tensor) -> torch.Tensor:
    """
    p: [B,1,H,W]
    return: [B,2] 1-based (x=col, y=row)
    """
    B, _, H, W = p.shape
    idx = p.view(B, -1).argmax(dim=1)
    r = (idx // W) + 1
    c = (idx % W) + 1
    return torch.stack([c, r], dim=1).float()

def to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return np.clip(x, 0, 1)

def overlay_heatmap(gray01: np.ndarray, heat01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    gray01: HxW in [0,1]
    heat01: HxW in [0,1]
    Returns RGB uint8 (heat injected into red channel for visibility)
    """
    base = np.stack([gray01, gray01, gray01], axis=2)  # RGB float
    out = base.copy()
    out[..., 0] = np.clip((1 - alpha) * out[..., 0] + alpha * heat01, 0, 1)  # red channel
    return (out * 255).astype(np.uint8)

def draw_cross_rgb(img_rgb_u8: np.ndarray, x: float, y: float, color_rgb, r: int = 6, t: int = 2) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    cx = int(round(x - 1))
    cy = int(round(y - 1))
    cx = max(0, min(W - 1, cx))
    cy = max(0, min(H - 1, cy))

    im = Image.fromarray(img_rgb_u8)
    dr = ImageDraw.Draw(im)
    # thickness via repeated lines
    for k in range(-(t//2), (t//2)+1):
        dr.line([(cx - r, cy + k), (cx + r, cy + k)], fill=tuple(color_rgb), width=1)
        dr.line([(cx + k, cy - r), (cx + k, cy + r)], fill=tuple(color_rgb), width=1)
    return np.array(im, dtype=np.uint8)


def draw_roi_box_rgb(img_rgb_u8: np.ndarray, x: float, y: float, box_w: int, box_h: int, color_rgb, t: int = 2) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    cx = int(round(x - 1))
    cy = int(round(y - 1))

    half_w = int(round(box_w / 2))
    half_h = int(round(box_h / 2))

    x0 = max(0, min(W - 1, cx - half_w))
    x1 = max(0, min(W - 1, cx + half_w))
    y0 = max(0, min(H - 1, cy - half_h))
    y1 = max(0, min(H - 1, cy + half_h))

    im = Image.fromarray(img_rgb_u8)
    dr = ImageDraw.Draw(im)
    # thickness by drawing multiple rectangles
    for k in range(t):
        dr.rectangle([x0 - k, y0 - k, x1 + k, y1 + k], outline=tuple(color_rgb))
    return np.array(im, dtype=np.uint8)


def canon_to_raw(xy_canon, orig_h: int, orig_w: int, canon_h: int, canon_w: int):
    """
    Map 1-based canonical coords -> 1-based original coords
    """
    x_c, y_c = float(xy_canon[0]), float(xy_canon[1])
    x_raw = x_c * (orig_w / float(canon_w))
    y_raw = y_c * (orig_h / float(canon_h))
    x_raw = float(np.clip(x_raw, 1.0, float(orig_w)))
    y_raw = float(np.clip(y_raw, 1.0, float(orig_h)))
    return x_raw, y_raw

# ============================================================
# Input loading
# ============================================================

def load_user_input(uploaded_file):
    """
    Supports:
      - .mat containing 'image' (HxW or HxWxT)
      - standard image files -> grayscale
    Optionally reads GT (if present):
      - centroid_corrected else centroid
    Returns:
      dict with img2d (float32), orig_h, orig_w, gt_xy_raw, note, n_frames
    """
    name = uploaded_file.name.lower()
    out = {"gt_xy_raw": None, "note": "", "n_frames": 1}

    if name.endswith(".mat"):
        S = sio.loadmat(uploaded_file)
        if "image" not in S:
            raise ValueError("MAT file must contain variable `image`.")
        img = S["image"]

        if img.ndim == 2:
            out["n_frames"] = 1
        elif img.ndim == 3:
            out["n_frames"] = int(img.shape[2])
        else:
            raise ValueError(f"`image` must be HxW or HxWxT, got shape={img.shape}")

        # Optional GT
        gt = None
        if "centroid_corrected" in S and S["centroid_corrected"] is not None:
            c = np.array(S["centroid_corrected"]).squeeze().astype(np.float32)
            if c.size >= 2:
                gt = (float(c[0]), float(c[1]))
        if gt is None and "centroid" in S and S["centroid"] is not None:
            c = np.array(S["centroid"]).squeeze().astype(np.float32)
            if c.size >= 2:
                gt = (float(c[0]), float(c[1]))

        out["mat_image"] = img.astype(np.float32)  # store full
        out["gt_xy_raw"] = gt
        out["note"] = f"Loaded MAT file with image shape {img.shape}."

        # pick frame later using slider (if T exists)
        if img.ndim == 2:
            img2d = img
        else:
            img2d = img[:, :, 0]
            out["note"] += " Default preview uses frame 0."

        img2d = img2d.astype(np.float32)
        out["orig_h"], out["orig_w"] = int(img2d.shape[0]), int(img2d.shape[1])
        out["img2d"] = img2d
        return out

    # image file
    im = Image.open(uploaded_file).convert("L")
    img2d = np.array(im).astype(np.float32)
    out["orig_h"], out["orig_w"] = int(img2d.shape[0]), int(img2d.shape[1])
    out["img2d"] = img2d
    out["note"] = f"Loaded image file ({out['orig_h']}x{out['orig_w']}) converted to grayscale."
    return out

# ============================================================
# Checkpoint handling (relative by default)
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_NAME = "best_centroid_xy_net_DSNT_EPE.pt"
DEFAULT_CKPT_PATH = APP_DIR / DEFAULT_CKPT_NAME

def resolve_ckpt_path(user_text: str) -> Path:
    """
    If user supplies a relative name, interpret it relative to app folder.
    If absolute, use it as-is.
    """
    p = Path(user_text.strip())
    if str(p) == "":
        return DEFAULT_CKPT_PATH
    if p.is_absolute():
        return p
    return (APP_DIR / p)

# ============================================================
# Model loading/inference
# ============================================================

@st.cache_resource
def load_model(ckpt_path_str: str, device: str):
    ckpt_path = str(ckpt_path_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint format unexpected. Expected a dict with key `'model'`.")

    model = TinyUNetLogits(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt

def run_inference(
    model,
    img2d: np.ndarray,
    canon_h=256,
    canon_w=256,
    beta=40.0,
    device="cpu",
    compute_peak=False,
    capture_features=False,
):
    # Resize to canonical space (PIL, no cv2)
    img_r = np.array(
        Image.fromarray(img2d.astype(np.float32)).resize(
            (canon_w, canon_h), resample=Image.BILINEAR
        ),
        dtype=np.float32,
    )

    # Robust normalization (match training)
    img_rn = robust_normalize(img_r)

    X = torch.from_numpy(img_rn).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    hook = None
    feats = None
    if capture_features:
        hook = FeatureHook(
            model, layer_names=["enc1.net", "enc2.net", "bot.net", "dec1.net", "out"]
        )

    with torch.no_grad():
        z = model(X)                             # logits [1,1,H,W]
        p = spatial_softmax_2d_logits(z, beta)   # prob   [1,1,H,W]

        xy = expected_xy_from_prob(p)[0].cpu().numpy()
        pk = hard_argmax_xy(p)[0].cpu().numpy() if compute_peak else None

        if capture_features and hook is not None:
            feats = {k: v.cpu() for k, v in hook.feats.items()}
            hook.close()

    z2d = z[0, 0].cpu().numpy().astype(np.float32)
    p2d = p[0, 0].cpu().numpy().astype(np.float32)

    xy_dsnt = (float(xy[0]), float(xy[1]))
    xy_peak = None if pk is None else (float(pk[0]), float(pk[1]))

    return img_r, z2d, p2d, xy_dsnt, xy_peak, feats



# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Interactive MRI Landmark Localization Teaching App", layout="wide")

st.title("Interactive MRI Landmark Localization Teaching App")
st.write(
    "Upload a **full-FOV** cine MRI image or a MATLAB `.mat` file containing `image` (HxW or HxWxT). "
    "Supported cine views include **2CH / 3CH / 4CH / SAX**, as long as the image is **not cropped**. "
    "The app visualizes the network probability map, DSNT centroid, and an ROI box centered at the centroid. "
    "**ROI size is automatically set to image size / 2** (scale-consistent across resolutions)."
)

with st.sidebar:
    st.header("Settings")

    # Default ckpt: relative in app folder
    ckpt_text = st.text_input(
        "Checkpoint path (.pt) [relative ok]",
        value=DEFAULT_CKPT_NAME if DEFAULT_CKPT_PATH.exists() else "",
        help="By default, the app searches for the checkpoint in the same folder as this app.",
    )

    canon_h = st.number_input("Canonical height", value=256, min_value=64, max_value=1024, step=16)
    canon_w = st.number_input("Canonical width", value=256, min_value=64, max_value=1024, step=16)

    beta = st.number_input(
        "Softmax beta (β)",
        value=40.0,
        min_value=1.0,
        max_value=200.0,
        step=1.0,
        help="β controls the sharpness of the probability map (NOT a threshold).",
    )
    alpha = st.slider("Heatmap overlay alpha", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    st.subheader("Toggles")
    show_peak = st.checkbox("Show highest-probability point (argmax)", value=False)
    show_net = st.checkbox("Show network structure", value=True)
    show_beta = st.checkbox("Show β (beta) explanation", value=True)
    show_torchinfo = st.checkbox("Show layer-by-layer summary (torchinfo)", value=False)
    st.subheader("Inside the network")
    show_features = st.checkbox("Show intermediate feature maps", value=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

uploaded = st.file_uploader("Upload an image or .mat", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mat"])

if uploaded is None:
    st.info("Upload an image or .mat to run the centroid predictor.")
    st.stop()

# Load input
try:
    data = load_user_input(uploaded)
except Exception as e:
    st.error("Failed to read file.")
    st.exception(e)
    st.stop()

# If MAT cine, allow frame selection
img2d = data["img2d"]
orig_h, orig_w = int(data["orig_h"]), int(data["orig_w"])
gt_xy_raw = data.get("gt_xy_raw", None)

st.info(data["note"])

if "mat_image" in data and data["mat_image"].ndim == 3:
    T = data["n_frames"]
    frame_idx = st.slider("Select cine frame (MAT only)", min_value=0, max_value=T - 1, value=0, step=1)
    img2d = data["mat_image"][:, :, frame_idx].astype(np.float32)
    orig_h, orig_w = int(img2d.shape[0]), int(img2d.shape[1])

# Teaching expanders (top)
if show_net:
    with st.expander("Network structure (TinyUNetLogits)", expanded=False):
        st.code(str(TinyUNetLogits(in_ch=1, base=32)), language="text")

if show_torchinfo:
    with st.expander("Layer-by-layer summary (optional: torchinfo)", expanded=False):
        try:
            from torchinfo import summary  # pip install torchinfo
            m = TinyUNetLogits(in_ch=1, base=32)
            s = summary(m, input_size=(1, 1, int(canon_h), int(canon_w)), verbose=0)
            st.text(str(s))
        except Exception as e:
            st.warning("torchinfo is not available or failed. Install with: pip install torchinfo")
            st.exception(e)

if show_beta:
    with st.expander("How β and DSNT work (simple explanation)"):
        st.code(
            "β is NOT a threshold.\n\n"
            "The network outputs logits z(x,y).\n\n"
            "Spatial softmax:\n"
            "p(x,y) = exp(β * z(x,y)) / sum_{u,v} exp(β * z(u,v))\n\n"
            "Effect of β:\n"
            "- Larger β  -> sharper probability map\n"
            "- Smaller β -> flatter probability map\n\n"
            "DSNT centroid (red cross):\n"
            "x_hat = sum_{x,y} p(x,y) * x\n"
            "y_hat = sum_{x,y} p(x,y) * y\n\n"
            "Argmax (yellow cross):\n"
            "Argmax = location of max p(x,y)\n\n"
            "No threshold is used. β only controls concentration."
        )


# Resolve checkpoint path (relative by default)
ckpt_path = resolve_ckpt_path(ckpt_text if ckpt_text.strip() else DEFAULT_CKPT_NAME)

# Checkpoint existence
if not ckpt_path.exists():
    st.error(
        f"Checkpoint not found: {ckpt_path}\n\n"
        f"Put `{DEFAULT_CKPT_NAME}` in the same folder as this app, or input a valid path."
    )
    st.stop()

# Run model + inference + visuals (guarded)
try:
    model, _ckpt = load_model(str(ckpt_path), device=device)

    img_canon, z2d, p2d, xy_dsnt, xy_peak, feats = run_inference(
        model, img2d,
        canon_h=int(canon_h), canon_w=int(canon_w),
        beta=float(beta), device=device,
        compute_peak=bool(show_peak),
        capture_features=True  # turn on
    )
    
    if show_features and feats is not None:
        st.divider()
        st.subheader("What did the network learn? (feature maps)")

        st.write(
            "Each block outputs multiple **channels** (feature maps). Early layers highlight edges/contrast; "
            "deeper layers become more heart-specific and suppress background. "
            "Below we visualize a few channels with the strongest activation."
        )

        def topk_channels(feat_t: torch.Tensor, k=8):
            # feat_t: [1,C,H,W]
            f = feat_t[0]  # [C,H,W]
            # pick channels with largest mean absolute activation
            scores = f.abs().mean(dim=(1,2))
            idx = torch.topk(scores, k=min(k, f.shape[0])).indices.tolist()
            return idx

        def tensor_to_u8(t: torch.Tensor):
            # t: [H,W]
            x = t.numpy().astype(np.float32)
            x = x - x.min()
            x = x / (x.max() + 1e-8)
            return (x * 255).clip(0,255).astype(np.uint8)

        for lname in ["enc1.net", "enc2.net", "bot.net", "dec1.net"]:
            if lname not in feats:
                continue
            ft = feats[lname]  # [1,C,H,W]
            idxs = topk_channels(ft, k=8)

            st.markdown(f"**Layer: `{lname}`**  (showing {len(idxs)} channels with strongest activation)")
            cols = st.columns(4)
            for i, ch in enumerate(idxs):
                fmap = ft[0, ch]
                img_u8 = tensor_to_u8(fmap)
                cols[i % 4].image(img_u8, caption=f"ch {ch}", use_container_width=True)

    # Convert predicted points to original coords
    xy_dsnt_raw = canon_to_raw(xy_dsnt, orig_h, orig_w, canon_h=int(canon_h), canon_w=int(canon_w))
    xy_peak_raw = None
    if xy_peak is not None:
        xy_peak_raw = canon_to_raw(xy_peak, orig_h, orig_w, canon_h=int(canon_h), canon_w=int(canon_w))

    # ROI definition: image size / 2
    roi_w_canon = int(canon_w) // 2
    roi_h_canon = int(canon_h) // 2
    roi_w_raw = orig_w // 2
    roi_h_raw = orig_h // 2

    # Visuals: canonical
    gray01 = to01(img_canon)
    heat01 = to01(p2d)

    canon_overlay = overlay_heatmap(gray01, heat01, alpha=float(alpha))
    canon_overlay = draw_cross_rgb(canon_overlay, xy_dsnt[0], xy_dsnt[1], color_rgb=(255, 0, 0))  # DSNT red
    canon_overlay = draw_roi_box_rgb(
        canon_overlay,
        xy_dsnt[0], xy_dsnt[1],
        roi_w_canon, roi_h_canon,
        color_rgb=(0, 255, 255), t=2
    )

    if xy_peak is not None:
        canon_overlay = draw_cross_rgb(canon_overlay, xy_peak[0], xy_peak[1], color_rgb=(255, 255, 0))  # argmax yellow

    prob_u8 = (heat01 * 255.0).clip(0, 255).astype(np.uint8)
    prob_rgb = np.stack([prob_u8, prob_u8, prob_u8], axis=2)

    # Visuals: original
    raw01 = to01(img2d)
    raw_rgb = (np.stack([raw01, raw01, raw01], axis=2) * 255).astype(np.uint8)

    raw_rgb = draw_cross_rgb(raw_rgb, xy_dsnt_raw[0], xy_dsnt_raw[1], color_rgb=(255, 0, 0))  # DSNT red
    raw_rgb = draw_roi_box_rgb(
        raw_rgb,
        xy_dsnt_raw[0], xy_dsnt_raw[1],
        roi_w_raw, roi_h_raw,
        color_rgb=(0, 255, 255), t=2
    )

    if xy_peak_raw is not None:
        raw_rgb = draw_cross_rgb(raw_rgb, xy_peak_raw[0], xy_peak_raw[1], color_rgb=(255, 255, 0))  # argmax yellow
    if gt_xy_raw is not None:
        raw_rgb = draw_cross_rgb(raw_rgb, gt_xy_raw[0], gt_xy_raw[1], color_rgb=(0, 255, 0))  # GT green

except Exception as e:
    st.error("Model inference failed.")
    st.exception(e)
    st.stop()

# ============================================================
# Results
# ============================================================

c1, c2 = st.columns([1, 1])

st.subheader("Canonical (model space): probability overlay + DSNT centroid + ROI (size = canon/2)")
st.image(canon_overlay, use_container_width=True)

st.subheader("Canonical probability map")
st.image(prob_rgb, caption="Brighter = higher probability.", use_container_width=True)


st.divider()
st.subheader("Predicted coordinates (1-based)")

st.write("**DSNT centroid (canonical x,y):**")
st.code(f"({xy_dsnt[0]:.2f}, {xy_dsnt[1]:.2f})")

st.write("**DSNT centroid (original x,y):**")
st.code(f"({xy_dsnt_raw[0]:.2f}, {xy_dsnt_raw[1]:.2f})")

if xy_peak is not None:
    st.write("**Argmax / peak (canonical x,y):**")
    st.code(f"({xy_peak[0]:.2f}, {xy_peak[1]:.2f})")

if xy_peak_raw is not None:
    st.write("**Argmax / peak (original x,y):**")
    st.code(f"({xy_peak_raw[0]:.2f}, {xy_peak_raw[1]:.2f})")

st.write("**ROI size (canonical):**")
st.code(f"{roi_w_canon} x {roi_h_canon}")

st.write("**ROI size (original):**")
st.code(f"{roi_w_raw} x {roi_h_raw}")

if gt_xy_raw is not None:
    dx = xy_dsnt_raw[0] - gt_xy_raw[0]
    dy = xy_dsnt_raw[1] - gt_xy_raw[1]
    epe = float(np.sqrt(dx * dx + dy * dy))
    st.write("**Ground truth centroid (original x,y):**")
    st.code(f"({gt_xy_raw[0]:.2f}, {gt_xy_raw[1]:.2f})")
    st.write("**EPE (DSNT vs GT) in original pixels:**")
    st.code(f"{epe:.2f} px")

with st.expander("What is β (softmax beta)?"):
    st.markdown("**β is not a threshold.** It controls how *peaked* the probability map becomes.")
    st.code(
        "p(x,y) = exp(β * z(x,y)) / sum_{u,v} exp(β * z(u,v))\n"
        "x_hat = sum_{x,y} p(x,y) * x\n"
        "y_hat = sum_{x,y} p(x,y) * y\n"
        "\n"
        "Argmax (yellow) = location of max p(x,y) (no threshold)."
    )

with st.expander("How the network turns features into a probability map"):
    st.write("**Logits** are the raw scores from the last convolution. Softmax(β) turns them into probabilities.")
    st.image(np.stack([(to01(z2d)*255).astype(np.uint8)]*3, axis=2),
             caption="Logits (higher = more likely centroid).", use_container_width=True)
    st.image(np.stack([(to01(p2d)*255).astype(np.uint8)]*3, axis=2),
             caption="Probability map after spatial softmax.", use_container_width=True)


