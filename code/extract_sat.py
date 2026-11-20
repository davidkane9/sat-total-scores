# /// script
# requires-python = ">=3.11"
# dependencies = ["pdfplumber==0.11.9"]
# ///
"""Parse each College Board "Total Group" SAT annual report PDF in
downloads/sat/total/ into the *raw* data files that the augmentation
step builds on:

    replicated/data/race/raw/<year>.csv     wide, one row per group
    replicated/data/race/raw/cohort_bands.csv long, cohort-level band counts
    replicated/data/moments/cohort.csv        long, per-year cohort moments

Covers graduating classes 2018-2026: the new SAT (400-1600 scale) with
the full per-race × score-band breakdown for the Total score (six
bands: 1400-1600, 1200-1390, ..., 400-590) on the "Score Distributions
by Subgroup" page.

Two report layouts are handled:

  * 2018-2025 — every number is printed as text in tables.

  * 2026       — redesigned report.  The per-race band breakdown is a
    stacked bar chart with only some segments labelled, so the band
    percentages are read from the widths of the bar rectangles (which
    are drawn at integer percents).  Race labels are shortened
    ("African American", "Two or More"), the ERW section is renamed
    RW, the cohort SD is no longer printed (so `sd` is blank for 2026
    in `moments/cohort.csv`), and the quartiles appear as a line of
    three scores above "25th percentile 50th percentile 75th
    percentile".

Output schemas
==============

Each per-year file `race/raw/<year>.csv` is *wide*, with one row per
group (7 named races + No Response + Total):

    race, N, mean, b_400_590, b_600_790, b_800_990,
    b_1000_1190, b_1200_1390, b_1400_1600

For the seven named races every column is filled with the integer
percent printed by the College Board.  For "No Response" and "Total"
the band columns are blank — the PDF does not print band percentages
for those rows.  The augmentation step (`code/augment_race.py`) fills
them in for No Response by subtraction; for Total it doesn't, because
the Total row's band info is delivered as absolute counts in
`cohort_bands.csv` rather than as percents.

`race/raw/cohort_bands.csv` carries the exact integer cohort band
counts for every year, long format:

    year, band, count

`moments/cohort.csv` carries the cohort-level five-number moments
(mean, SD, three quartile scores) per year, long format:

    year, mean, sd, p25, p50, p75

What's deliberately *not* here
==============================

This script does not derive the No Response band percentages.  Today
that derivation is a separate step (`code/augment_race.py`) that reads
this file plus `cohort_bands.csv` and writes a long-format
`race/augmented/<year>.csv` that downstream consumers read.  Keeping
"what came from the PDF" and "what was computed afterwards" in two
different files makes hand-editing of the PDF-sourced data safe: a
human can edit `race/raw/<year>.csv` and re-run the augmentation step
without re-running this script (which is the only thing that requires
the source PDFs).

Run:
    python3 code/extract_sat.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR     = ROOT / "downloads" / "sat" / "total"
RAW_DIR     = ROOT / "replicated" / "data" / "race" / "raw"
MOMENTS_DIR = ROOT / "replicated" / "data" / "moments"

# Race rows on the per-race summary page, in print order (also the order
# we write them into the raw CSV).  This matches the "Race / Ethnicity"
# section header order.
NEW_RACE_LABELS = [
    "American Indian/Alaska Native",
    "Asian",
    "Black/African American",
    "Hispanic/Latino",
    "Native Hawaiian/Other Pacific Islander",
    "White",
    "Two or More Races",
    "No Response",
]

# Score bands in the per-race distribution table on the "Score
# Distributions by Subgroup" page, in PDF print order.  We rewrite to
# the file/CSV-friendly ascending order (low to high) on output.
NEW_BANDS_PDF = ["1400–1600", "1200–1390", "1000–1190", "800–990", "600–790", "400–590"]
BAND_ORDER    = ["400-590", "600-790", "800-990", "1000-1190", "1200-1390", "1400-1600"]
BAND_COL      = {b: f"b_{b.replace('-', '_')}" for b in BAND_ORDER}

# Column headers in the score-distribution table (PDF print order).
# We keep only the seven race columns; the Total/Female/Male columns
# are aggregations or splits we don't need.
DIST_COLUMNS = [
    "Total Students", "Female", "Male",
    "American Indian", "Asian", "African American", "Hispanic",
    "Native Hawaiian", "White", "Two or More Races",
]
# Map distribution-table column header → race summary label so the two
# tables join cleanly.
DIST_TO_SUMMARY_RACE = {
    "American Indian":   "American Indian/Alaska Native",
    "Asian":             "Asian",
    "African American":  "Black/African American",
    "Hispanic":          "Hispanic/Latino",
    "Native Hawaiian":   "Native Hawaiian/Other Pacific Islander",
    "White":             "White",
    "Two or More Races": "Two or More Races",
}


def parse_race_summary(page_text: str) -> dict[str, tuple[int, int]]:
    """Pull (N, mean_total) for each of the eight race rows on the
    per-race summary page.  Lines look like
        American Indian/Alaska Native 9,237 0% 874 486 477 27% 53% 29% 45%
    Columns: Race | Number | Percent | Total | ERW | Math | Both | ERW% | Math% | None%.
    Returns {race_label: (N, mean_total)}.  Missing labels are silently
    skipped — caller may want to assert completeness."""
    out: dict[str, tuple[int, int]] = {}
    for label in NEW_RACE_LABELS:
        pat = (
            rf"^\s*{re.escape(label)}\s+"
            r"([\d,]+)\s+"            # N
            r"\d+%\s+"                # Percent of cohort
            r"(\d+)\s+"               # Mean total
            r"\d+\s+\d+\s+"           # ERW mean, Math mean
            r"\d+%\s+\d+%\s+\d+%\s+\d+%"
        )
        m = re.search(pat, page_text, re.MULTILINE)
        if m:
            out[label] = (int(m.group(1).replace(",", "")), int(m.group(2)))
    return out


def parse_cohort_summary(page_text: str) -> tuple[int, int] | None:
    """Pull (N, mean_total) for the cohort "Total" row on the per-race
    summary page. Format example (2025): "Total 2,004,965 1029 521 508 39% 64% 41% 34%"."""
    m = re.search(
        r"^\s*Total\s+([\d,]+)\s+(\d+)\s+\d+\s+\d+\s+\d+%\s+\d+%\s+\d+%\s+\d+%",
        page_text, re.MULTILINE)
    if not m:
        return None
    return (int(m.group(1).replace(",", "")), int(m.group(2)))


def parse_score_band_distribution(page_text: str) -> dict[tuple[str, str], int]:
    """Pull seven named-race × six band percentages from the
    'Score Distributions by Subgroup' page (Total Score subsection).
    Returns {(race_label, band): percent}."""
    out: dict[tuple[str, str], int] = {}
    m_block = re.search(
        r"Total Score\s*\n(.*?)(?:Section Scores|Section \(Test\))",
        page_text, re.DOTALL)
    if not m_block:
        return out
    block = m_block.group(1)
    for band_pdf in NEW_BANDS_PDF:
        pat = re.escape(band_pdf) + r"\s+" + r"\s+".join([r"(\d+)%"] * 10)
        m = re.search(pat, block)
        if not m:
            continue
        pcts = [int(g) for g in m.groups()]
        band = band_pdf.replace("–", "-")
        for col_idx, col in enumerate(DIST_COLUMNS):
            if col in DIST_TO_SUMMARY_RACE:
                out[(DIST_TO_SUMMARY_RACE[col], band)] = pcts[col_idx]
    return out


def parse_cohort_band_counts(page_text: str) -> dict[str, int]:
    """Pull the cohort Total-Score band counts from the 'Total and
    Section Scores' page. Lines look like
        1400–1600 149,767 7% 700–800 170,667 9% 185,306 9%
    The first count after the score range is the Total Score count;
    the rest are ERW/Math section breakdowns we don't keep.
    Returns {band: cohort_count}."""
    out: dict[str, int] = {}
    pat = re.compile(
        r"^\s*"
        r"(1400|1200|1000|800|600|400)"
        r"[–-]\s*"
        r"(?:1600|1390|1190|990|790|590)"
        r"\s+([\d,]+)"
    )
    ranges = {1400: "1400-1600", 1200: "1200-1390", 1000: "1000-1190",
              800:  "800-990",   600:  "600-790",   400:  "400-590"}
    for line in page_text.split("\n"):
        m = pat.match(line)
        if not m:
            continue
        lo = int(m.group(1))
        if lo in ranges and ranges[lo] not in out:
            out[ranges[lo]] = int(m.group(2).replace(",", ""))
    return out


def parse_cohort_sd(page_text: str) -> int | None:
    """Cohort Total Score SD from the 'Total and Section Scores' page.
    Line: 'SD 235 SD 121 121' or 'SD 235' alone. We take the first SD."""
    m = re.search(r"^\s*SD\s+(\d+)", page_text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_cohort_percentiles(page_text: str) -> dict[int, int]:
    """Cohort Total-Score 25th/50th/75th percentiles from the 'SAT
    Suite Performance: Interquartile Ranges' page.  Each percentile
    row starts e.g. '75th 1210 610 600 ...' (Total, ERW, Math, ...);
    the first occurrence per percentile is the SAT row (PSAT sub-tables
    follow). Returns {25: ..., 50: ..., 75: ...} for whatever was found."""
    out: dict[int, int] = {}
    for pctl in (25, 50, 75):
        m = re.search(rf"^\s*{pctl}th\s+(\d+)\b", page_text, re.MULTILINE)
        if m:
            out[pctl] = int(m.group(1))
    return out


# ─────────────────────────────────────────────────────────────────────
# 2026+ layout: short race labels, bar-chart subgroup distributions
# ─────────────────────────────────────────────────────────────────────
SHORT_TO_SUMMARY_RACE = {
    "American Indian":  "American Indian/Alaska Native",
    "Asian":            "Asian",
    "African American": "Black/African American",
    "Hispanic":         "Hispanic/Latino",
    "Native Hawaiian":  "Native Hawaiian/Other Pacific Islander",
    "White":            "White",
    "Two or More":      "Two or More Races",
    "No Response":      "No Response",
}


def parse_v3_race_summary(page_text: str) -> dict[str, tuple[int, int]]:
    """2026+ per-race summary rows, e.g.
        African American 233,864 12% 917 470 447 20% 46% 22% 52%
    Returns {race_label: (N, mean_total)} keyed by the full 2018-2025
    labels so every year's raw file uses the same names."""
    out: dict[str, tuple[int, int]] = {}
    for short, label in SHORT_TO_SUMMARY_RACE.items():
        pat = (rf"^\s*{re.escape(short)}\s+([\d,]+)\s+\d+%\s+(\d+)\s+\d+\s+\d+\s+"
               r"\d+%\s+\d+%\s+\d+%\s+\d+%")
        m = re.search(pat, page_text, re.MULTILINE)
        if m:
            out[label] = (int(m.group(1).replace(",", "")), int(m.group(2)))
    return out


def parse_v3_band_chart(page, year: int) -> dict[tuple[str, str], int]:
    """Read the per-race Total-score band percentages from the stacked
    bar chart on the 2026+ 'Score Distributions by Subgroup' page.

    Each race row is six rectangles (drawn twice each) whose widths are
    integer percents times a common scale.  A band with 0% has no
    rectangle, so segments are matched to bands by fill colour, using a
    row that has all six as the reference.  The scale is the full width
    of such a row.  Returns {(race_label, band): percent}."""
    words = page.extract_words()
    # Row anchors: the first word of each race label in the
    # Race/Ethnicity chart.
    first_words = {"American": "American Indian", "Asian": "Asian",
                   "African": "African American", "Hispanic": "Hispanic",
                   "Native": "Native Hawaiian", "White": "White",
                   "Two": "Two or More"}
    hdr = [w for w in words if w["text"] == "Race/Ethnicity"]
    if not hdr:
        return {}
    y_hdr = hdr[0]["top"]
    anchors: dict[str, float] = {}
    for w in words:
        if w["text"] in first_words and w["x0"] < 100 and w["top"] > y_hdr:
            anchors.setdefault(first_words[w["text"]], w["top"])

    def row_rects(top):
        """Rectangles on one race's row, deduplicated, left to right."""
        seen, out = set(), []
        for r in page.rects:
            if abs(r["top"] - top) < 8 and r["x0"] > 120 and r["width"] > 0.5:
                key = (round(r["x0"], 1), round(r["width"], 1))
                if key not in seen:
                    seen.add(key)
                    out.append(r)
        return sorted(out, key=lambda r: r["x0"])

    rects = {short: row_rects(top) for short, top in anchors.items()}
    full = [rs for rs in rects.values() if len(rs) == 6]
    if not full:
        return {}
    colours = [str(r.get("non_stroking_color")) for r in full[0]]
    scale = (full[0][-1]["x1"] - full[0][0]["x0"]) / 100.0
    out: dict[tuple[str, str], int] = {}
    for short, rs in rects.items():
        pct = [0] * 6
        for r in rs:
            c = str(r.get("non_stroking_color"))
            if c not in colours:
                raise ValueError(f"{year} {short}: unmatched segment colour {c}")
            pct[colours.index(c)] = round(r["width"] / scale)
        if not 98 <= sum(pct) <= 102:
            raise ValueError(f"{year} {short}: band percents sum to {sum(pct)}")
        for b, band in enumerate(BAND_ORDER):
            out[(SHORT_TO_SUMMARY_RACE[short], band)] = pct[b]
    return out


def parse_v3_percentiles(page_text: str) -> dict[int, int]:
    """2026+: the three Total-score quartiles are printed above their
    labels, so the extracted text has a line '870 1030 1220' followed by
    '25th percentile 50th percentile 75th percentile'."""
    lines = page_text.splitlines()
    for i, line in enumerate(lines[1:], start=1):
        if re.search(r"25th percentile\s+50th percentile\s+75th percentile", line):
            m = re.match(r"\s*(\d{3,4})\s+(\d{3,4})\s+(\d{3,4})\s*$", lines[i - 1])
            if m:
                return {25: int(m.group(1)), 50: int(m.group(2)), 75: int(m.group(3))}
    return {}


def extract_year_v3(pdf, page_texts: list[str], year: int):
    """2026+ report layout.  Same return shape as extract_year; the
    cohort SD is always None because the report no longer prints it."""
    race_summary: dict[str, tuple[int, int]] = {}
    cohort_summary: tuple[int, int] | None = None
    band_pcts: dict[tuple[str, str], int]   = {}
    cohort_bands: dict[str, int]            = {}
    cohort_pct: dict[int, int]              = {}

    for txt in page_texts:
        if re.search(r"^\s*African American\s+[\d,]+\s+\d+%", txt, re.MULTILINE):
            race_summary   = parse_v3_race_summary(txt)
            cohort_summary = parse_cohort_summary(txt)
            break
    # The cohort Total row sits on the participation page, before the
    # per-race page.  Take the first page that has it (the SAT pages
    # precede the PSAT pages).
    if cohort_summary is None:
        for txt in page_texts:
            cohort_summary = parse_cohort_summary(txt)
            if cohort_summary is not None:
                break
    for page, txt in zip(pdf.pages, page_texts):
        if "Score Distributions by Subgroup" in txt and "Race/Ethnicity" in txt:
            band_pcts = parse_v3_band_chart(page, year)
            break
    for txt in page_texts:
        if "Total Scores" in txt and re.search(r"1400[–-]1600\s+[\d,]+", txt):
            cohort_bands = parse_cohort_band_counts(txt)
            cohort_pct   = parse_v3_percentiles(txt)
            break

    return race_summary, cohort_summary, band_pcts, cohort_bands, None, cohort_pct


def extract_year(pdf_path: Path):
    """Return (race_summary, cohort_summary, band_pcts, cohort_bands,
    cohort_sd, cohort_pct) for one PDF.  See call site for the schemas."""
    year = int(pdf_path.stem)
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [p.extract_text() or "" for p in pdf.pages]
        if year >= 2026:
            return extract_year_v3(pdf, page_texts, year)

    race_summary: dict[str, tuple[int, int]] = {}
    cohort_summary: tuple[int, int] | None = None
    band_pcts: dict[tuple[str, str], int]   = {}
    cohort_bands: dict[str, int]            = {}
    cohort_sd: int | None                   = None
    cohort_pct: dict[int, int]              = {}

    for txt in page_texts:
        if "Race / Ethnicity" in txt and "ERW" in txt and "Math" in txt:
            race_summary  = parse_race_summary(txt)
            cohort_summary = parse_cohort_summary(txt)
            break
    for txt in page_texts:
        if "Score Distributions by Subgroup" in txt and "Total Score" in txt:
            band_pcts = parse_score_band_distribution(txt)
            break
    for txt in page_texts:
        if (re.search(r"Total\s+Score.*\bERW\b.*\bMath\b", txt) and
                re.search(r"1400[–-]1600\s+[\d,]+", txt)):
            cohort_bands = parse_cohort_band_counts(txt)
            cohort_sd    = parse_cohort_sd(txt)
            break
    for txt in page_texts:
        if ("Interquartile" in txt and
                re.search(r"^\s*75th\s+\d+", txt, re.MULTILINE)):
            cohort_pct = parse_cohort_percentiles(txt)
            break

    return race_summary, cohort_summary, band_pcts, cohort_bands, cohort_sd, cohort_pct


def write_raw_csv(year: int, race_summary: dict[str, tuple[int, int]],
                  cohort_summary: tuple[int, int] | None,
                  band_pcts: dict[tuple[str, str], int]) -> Path:
    """Wide per-year file: one row per group, columns race/N/mean +
    six band-percent columns.  No Response and Total rows have blank
    band columns (the PDF doesn't print percentages for them)."""
    out = RAW_DIR / f"{year}.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["race", "N", "mean"] + [BAND_COL[b] for b in BAND_ORDER]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for race in NEW_RACE_LABELS:
            if race not in race_summary:
                continue
            n, mean = race_summary[race]
            row = {"race": race, "N": n, "mean": mean}
            for band in BAND_ORDER:
                row[BAND_COL[band]] = band_pcts.get((race, band), "")
            w.writerow(row)
        if cohort_summary is not None:
            n, mean = cohort_summary
            row = {"race": "Total", "N": n, "mean": mean}
            for band in BAND_ORDER:
                row[BAND_COL[band]] = ""
            w.writerow(row)
    return out


def write_cohort_bands_csv(records: list[dict]) -> Path:
    """Long file (one row per (year, band)) holding the integer cohort
    band counts that augment_race.py needs to back out No Response by
    subtraction."""
    out = RAW_DIR / "cohort_bands.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "band", "count"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    return out


def write_moments_csv(records: list[dict]) -> Path:
    out = MOMENTS_DIR / "cohort.csv"
    MOMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "mean", "sd", "p25", "p50", "p75"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    return out


def main() -> None:
    moments: list[dict] = []
    cohort_band_records: list[dict] = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        year = int(pdf_path.stem)
        race_summary, cohort_summary, band_pcts, cohort_bands, cohort_sd, cohort_pct = \
            extract_year(pdf_path)
        out = write_raw_csv(year, race_summary, cohort_summary, band_pcts)
        n_rows = len(race_summary) + (1 if cohort_summary else 0)
        print(f"{year}: {n_rows} rows → {out.relative_to(ROOT)}")
        for band in BAND_ORDER:
            if band in cohort_bands:
                cohort_band_records.append(
                    {"year": year, "band": band, "count": cohort_bands[band]})
        cohort_mean = cohort_summary[1] if cohort_summary else None
        if cohort_mean is not None:
            moments.append({"year": year, "mean": cohort_mean,
                            "sd":  cohort_sd if cohort_sd is not None else "",
                            "p25": cohort_pct.get(25, ""),
                            "p50": cohort_pct.get(50, ""),
                            "p75": cohort_pct.get(75, "")})
    if cohort_band_records:
        cout = write_cohort_bands_csv(cohort_band_records)
        print(f"\nWrote cohort band counts → {cout.relative_to(ROOT)}")
    if moments:
        mout = write_moments_csv(moments)
        print(f"Wrote cohort moments → {mout.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
