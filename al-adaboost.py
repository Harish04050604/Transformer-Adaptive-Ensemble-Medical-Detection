import os

HAM_DIR = "unlabelled/softmax"
RAM_DIR = "unlabelled/softlabels"
OUTPUT_DIR = "unlabelled/class_prediction"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_score(path):
    with open(path, "r") as f:
        line = f.readline().strip()

        if line == "" or line == "N/A":
            return None

        parts = line.split()

        try:
            if len(parts) == 1:
                return float(parts[0])
            elif len(parts) >= 2:
                return float(parts[1])
        except:
            return None

    return None


for file in os.listdir(HAM_DIR):

    ham_path = os.path.join(HAM_DIR, file)
    ram_path = os.path.join(RAM_DIR, file)

    if not os.path.exists(ram_path):
        continue

    p_ham = read_score(ham_path)
    p_ram = read_score(ram_path)

    # =========================
    # CASE HANDLING
    # =========================

    # Case 1: both missing
    if p_ham is None and p_ram is None:
        print(f"Skipping {file} (both N/A)")
        continue

    # Case 2: only HAM exists
    elif p_ram is None:
        p_cls = p_ham
        print(f"{file} → using HAM only: {p_cls:.4f}")

    # Case 3: only RAM exists
    elif p_ham is None:
        p_cls = p_ram
        print(f"{file} → using RAM only: {p_cls:.4f}")

    # Case 4: both exist → AdaBoost
    else:
        agreement = abs(p_ham - p_ram)

        w_ham = (1 - agreement) * p_ham
        w_ram = (1 - agreement) * p_ram

        total = w_ham + w_ram
        if total == 0:
            w_ham, w_ram = 0.5, 0.5
        else:
            w_ham /= total
            w_ram /= total

        p_cls = w_ham * p_ham + w_ram * p_ram

        print(f"{file} → fused: {p_cls:.4f}")

    # =========================
    # SAVE OUTPUT
    # =========================
    out_path = os.path.join(OUTPUT_DIR, file)

    with open(out_path, "w") as f:
        f.write(f"0 {p_cls:.4f}\n")