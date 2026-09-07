import re, sys, urllib.request, os
BASE = "https://mirrors.aliyun.com/pypi/simple"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
op.addheaders = [("User-Agent", "pip/24.0")]
BAD = re.compile(r"(riscv64|aarch64|armv7l|i686|ppc64|s390x|win32|win_amd64|macosx|musllinux)")
out = []
for line in open("/tmp/pkglist.txt"):
    line = line.strip()
    if not line or "==" not in line: continue
    name, ver = line.split("==", 1)
    norm = re.sub(r"[-_.]+", "-", name).lower()
    try:
        html = op.open(f"{BASE}/{norm}/", timeout=30).read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"# FAIL {name}: {e}", file=sys.stderr); continue
    cands = []
    for h in re.findall(r'href="([^"#]+)', html):
        fn = h.split("/")[-1]
        if not fn.endswith(".whl") or f"-{ver}-" not in fn or BAD.search(fn):
            continue
        py_ok = ("cp312" in fn) or ("py3-none" in fn) or ("abi3" in fn and "cp3" in fn) or ("py2.py3" in fn)
        if not py_ok: continue
        plat_ok = fn.endswith("-any.whl") or "x86_64" in fn
        if not plat_ok: continue
        # abi3 wheels must be for cp312 or lower
        m = re.search(r"cp3(\d+)-abi3", fn)
        if m and int(m.group(1)) > 12: continue
        cands.append((("manylinux_2_28" in fn), ("manylinux" in fn), ("cp312" in fn), h))
    if not cands:
        print(f"# NOWHEEL {name}=={ver}", file=sys.stderr); continue
    h = sorted(cands)[-1][-1]
    if h.startswith("../../"): h = "https://mirrors.aliyun.com/pypi/" + h[6:]
    elif h.startswith("/"):    h = "https://mirrors.aliyun.com" + h
    out.append(h)
print("\n".join(out))
