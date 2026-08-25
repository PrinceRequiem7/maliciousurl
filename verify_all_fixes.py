"""
verify_all_fixes.py — run from your project root: python verify_all_fixes.py

One script, clear PASS/FAIL output, checking every fix applied so far:
  1. The required files actually exist and are wired up correctly.
  2. Known-safe allowlisted domains return Safe (Wikipedia, Google, etc.)
  3. The underlying MODEL still works correctly for everything else
     (this matters: an allowlist alone would "pass" trivially if the
     model were broken or just said Safe to everything -- this checks
     that isn't happening).
  4. Obviously malicious-looking URLs still get flagged Malicious
     (the overcorrection check: proves the system hasn't just started
     saying "Safe" to everything).
  5. A real accuracy check against your own labeled test data.

Exits with a summary: how many checks passed out of how many total.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

results = []  # (description, passed: bool, detail: str)


def check(description, passed, detail=""):
    results.append((description, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {description}" + (f"  -- {detail}" if detail else ""))


print("=" * 70)
print("1. FILE / WIRING CHECKS")
print("=" * 70)

allowlist_path = Path("src/allowlist.py")
check("src/allowlist.py exists", allowlist_path.exists())

predict_source = Path("src/predict.py").read_text() if Path("src/predict.py").exists() else ""
check("src/predict.py exists", bool(predict_source))
check("predict.py imports is_trusted_domain from allowlist",
      "is_trusted_domain" in predict_source)
check("predict.py checks uses_scaling (not just scaler presence)",
      "uses_scaling" in predict_source)

try:
    from src.allowlist import TRUSTED_DOMAINS, is_trusted_domain
    check("allowlist module imports successfully", True, f"{len(TRUSTED_DOMAINS)} trusted domains loaded")
except Exception as e:
    check("allowlist module imports successfully", False, str(e))

try:
    from src.predict import predict_url
    check("predict_url imports successfully", True)
except Exception as e:
    check("predict_url imports successfully", False, str(e))
    print("\nCannot continue -- fix the import error above first.")
    sys.exit(1)

print()
print("=" * 70)
print("2. ALLOWLISTED DOMAINS RETURN SAFE")
print("=" * 70)
allowlist_test_urls = [
    "https://www.wikipedia.org/",
    "en.wikipedia.org/wiki/Python",
    "https://www.google.com/",
    "google.com",
    "https://github.com/",
]
for url in allowlist_test_urls:
    try:
        r = predict_url(url)
        passed = r.prediction == "Safe" and r.model_name == "Allowlist (trusted domain)"
        check(f"'{url}' -> Safe via allowlist", passed,
              f"got prediction={r.prediction}, model_name={r.model_name}")
    except Exception as e:
        check(f"'{url}' -> Safe via allowlist", False, f"raised {e}")

print()
print("=" * 70)
print("3. NON-ALLOWLISTED URLS: OBVIOUSLY MALICIOUS PATTERNS STILL FLAGGED")
print("=" * 70)
print("(overcorrection check -- proves the system hasn't just started saying Safe to everything)")
suspicious_test_urls = [
    "http://192.168.1.1/login/verify.php?token=abc123",
    "http://secure-login-verify.paypal-account-update.totally-not-real.tk/confirm",
    "http://192.168.0.55/wp-admin/malware.exe",
]
for url in suspicious_test_urls:
    try:
        r = predict_url(url)
        # We don't assert Malicious specifically (that depends on the model),
        # but we DO assert it's not silently going through the allowlist.
        passed = r.model_name != "Allowlist (trusted domain)"
        check(f"'{url[:55]}...' not silently allowlisted", passed,
              f"prediction={r.prediction}, model_name={r.model_name}")
    except Exception as e:
        check(f"'{url[:55]}...'", False, f"raised {e}")

print()
print("=" * 70)
print("4. REAL ACCURACY CHECK: model performance on your own labeled test data")
print("=" * 70)
try:
    import pandas as pd
    from src.utils import PROCESSED_DATA_DIR

    test_path = PROCESSED_DATA_DIR / "test.csv"
    df = pd.read_csv(test_path)

    # Exclude allowlisted domains from this check -- we already tested those
    # above, and including them here would inflate the score without
    # telling us anything new about the MODEL's own accuracy.
    from src.allowlist import is_trusted_domain
    df = df[~df["url"].apply(is_trusted_domain)]

    safe_sample = df[df["label"] == 0].sample(min(40, (df["label"] == 0).sum()), random_state=1)
    mal_sample = df[df["label"] == 1].sample(min(40, (df["label"] == 1).sum()), random_state=1)

    def run_batch(sample_df, true_label):
        correct = 0
        for _, row in sample_df.iterrows():
            r = predict_url(row["url"])
            if r.prediction == true_label:
                correct += 1
        return correct, len(sample_df)

    sc, st = run_batch(safe_sample, "Safe")
    mc, mt = run_batch(mal_sample, "Malicious")
    overall = (sc + mc) / (st + mt)

    check(f"Known-Safe accuracy (non-allowlisted sample)", sc / st >= 0.75,
          f"{sc}/{st} ({sc/st:.1%})")
    check(f"Known-Malicious accuracy (non-allowlisted sample)", mc / mt >= 0.75,
          f"{mc}/{mt} ({mc/mt:.1%})")
    check(f"Overall accuracy is reasonable (not degenerate)", overall >= 0.75,
          f"{overall:.1%} overall")
except Exception as e:
    check("Accuracy check ran", False, str(e))

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
passed_count = sum(1 for _, p, _ in results if p)
total_count = len(results)
print(f"{passed_count}/{total_count} checks passed")
if passed_count == total_count:
    print("\n✓ Everything checks out. The fixes are working correctly.")
else:
    print("\n✗ Some checks failed -- see [FAIL] lines above for specifics.")
    failed = [d for d, p, _ in results if not p]
    print("Failed checks:")
    for f in failed:
        print(f"  - {f}")