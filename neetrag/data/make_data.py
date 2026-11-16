"""Generate SYNTHETIC cut-off data. Every college and number here is invented.
Run: python data/make_data.py  (deterministic, seed fixed)."""
import csv, random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).parent

# (name, state, type, base General closing rank for MBBS in 2025, has BDS)
COLLEGES = [
    ("Sundarpur Government Medical College", "West Bengal", "Government", 900, True),
    ("Rajghat Government Medical College", "West Bengal", "Government", 2600, False),
    ("Nilgiri Government Medical College", "Tamil Nadu", "Government", 1500, True),
    ("Kaveri Government Medical College", "Tamil Nadu", "Government", 4200, False),
    ("Aravali Government Medical College", "Rajasthan", "Government", 3100, True),
    ("Thar Government Medical College", "Rajasthan", "Government", 7800, False),
    ("Gomti Government Medical College", "Uttar Pradesh", "Government", 2100, False),
    ("Vindhya Government Medical College", "Madhya Pradesh", "Government", 5400, True),
    ("Lakeview Private Medical College", "Karnataka", "Private", 21000, True),
    ("Harbour Private Medical College", "Maharashtra", "Private", 34000, False),
    ("Meridian Private Medical College", "Karnataka", "Private", 52000, True),
    ("Orchid Private Medical College", "West Bengal", "Private", 61000, False),
    ("Summit Private Medical College", "Tamil Nadu", "Private", 78000, True),
    ("Pinewood Private Medical College", "Maharashtra", "Private", 96000, False),
]
CAT_FACTOR = {"General": 1.0, "EWS": 1.35, "OBC": 1.7, "SC": 5.5, "ST": 9.0}
YEAR_DRIFT = {2023: 0.88, 2024: 0.95, 2025: 1.0}
SEATS = {"Government": 150, "Private": 100}

rows = []
for i, (name, state, typ, base, bds) in enumerate(COLLEGES, 1):
    for course, cfac in [("MBBS", 1.0)] + ([("BDS", 6.0)] if bds else []):
        for cat, f in CAT_FACTOR.items():
            for year, d in YEAR_DRIFT.items():
                rank = int(base * cfac * f * d * random.uniform(0.93, 1.07))
                rows.append([name, state, typ, course, cat, year, rank,
                             SEATS[typ] if course == "MBBS" else 50])
for n, r in enumerate(rows, 1):
    r.insert(0, f"R{n:03d}")
with open(OUT / "cutoffs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "college", "state", "type", "course", "category",
                "year", "closing_rank", "seats"])
    w.writerows(rows)

NOTES = [
    "Closing rank is the rank of the last candidate allotted a seat in the final round. A candidate with a rank equal to or better (lower number) than the closing rank would have received that seat in that year.",
    "Past closing ranks are not predictions. Ranks move every year with the number of candidates, seats added and changes in reservation policy.",
    "Reserved-category seats (OBC, SC, ST, EWS) use the candidate's category rank, not the all-India rank. General seats use the all-India rank.",
    "State quota seats in Government colleges are normally open only to candidates domiciled in that state. All-India quota seats are open to all candidates.",
    "Private colleges charge tuition fees set by the college or state committee. This dataset does not contain fee information.",
    "This dataset does not contain hostel, stipend, bond or placement information. Questions about these are outside the sources.",
    "Data covers the years 2023, 2024 and 2025, final round only. There is no data for any other year.",
    "All colleges, ranks and seat counts in this dataset are synthetic and invented for a technical assignment. They do not describe any real institution.",
]
with open(OUT / "notes.md", "w") as f:
    for n, t in enumerate(NOTES, 1):
        f.write(f"[N{n}] {t}\n\n")
print(len(rows), "cut-off rows,", len(NOTES), "notes")
