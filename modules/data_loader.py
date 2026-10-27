"""
data_loader.py
Reads uploaded files safely, detects column types, and checks data quality.
Nothing here changes the user's original data; it only reads and reports.
"""
from __future__ import annotations

import io
import os
import re
import warnings

import pandas as pd
from pandas.api import types as ptypes

MAX_FILE_MB = 25
ALLOWED_EXTENSIONS = (".csv", ".xls", ".xlsx")
EXCEL_EXTENSIONS = (".xls", ".xlsx")

# Column-name words that suggest "this is an identifier, not a measurement"
ID_TOKENS = {"id", "code", "sku", "ref", "reference", "uuid", "number", "no", "num", "key"}
ID_FIRST_TOKENS = {"id", "uuid"}
ID_COMPACT_SUFFIXES = (
    "customerid", "productid", "orderid", "userid",
    "clientid", "itemid", "invoiceid", "transactionid",
)

# Text that looks like a date (e.g. 2025-01-31, 31/01/2025, 5 Jan 2025, Jan 5, 2025)
DATE_PATTERN = (
    r"\d{4}[-/]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}"
)

# Currency words/symbols, commas, spaces, and % are removed before reading numbers
CURRENCY_PATTERN = r"(?i)rwf|frw|usd|eur|gbp|kes|ugx|tzs|[$€£,%\s]"


class DataLoadError(Exception):
    """A problem with the uploaded file, written in plain English for the user."""


# --------------------------------------------------------------------------
# File validation and loading
# --------------------------------------------------------------------------
def get_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def is_excel(filename: str) -> bool:
    return get_extension(filename) in EXCEL_EXTENSIONS


def validate_file(filename: str, size_bytes: int) -> str:
    """Check type and size. Returns the file extension if the file is acceptable."""
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(
            f"'{filename}' is not a supported file type. "
            "Please upload a CSV, XLS, or XLSX file."
        )
    if size_bytes == 0:
        raise DataLoadError(f"'{filename}' is empty (0 bytes). Please upload a file that contains data.")
    if size_bytes > MAX_FILE_MB * 1024 * 1024:
        raise DataLoadError(
            f"'{filename}' is larger than the {MAX_FILE_MB} MB limit. "
            "Please upload a smaller file or split it into parts."
        )
    return ext


def _excel_engine(ext: str) -> str:
    return "openpyxl" if ext == ".xlsx" else "xlrd"


def get_excel_sheet_names(file_bytes: bytes, filename: str) -> list[str]:
    ext = get_extension(filename)
    try:
        with pd.ExcelFile(io.BytesIO(file_bytes), engine=_excel_engine(ext)) as xls:
            return list(xls.sheet_names)
    except ImportError:
        raise DataLoadError("A required library for this Excel format is missing. Run: pip install -r requirements.txt")
    except Exception:
        raise DataLoadError(
            f"Could not open '{filename}'. The Excel file may be corrupted, "
            "password-protected, or not a real Excel file."
        )


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    if b"\x00" in file_bytes[:4096]:
        raise DataLoadError("This file looks like a binary file, not a CSV. Please check that you chose the right file.")

    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # sep=None lets pandas guess the separator (comma, semicolon, tab...)
            return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise DataLoadError("The CSV file is empty. Please upload a file that contains data.")
        except Exception as first_error:
            last_error = first_error
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise DataLoadError("The CSV file is empty. Please upload a file that contains data.")
            except Exception as second_error:
                last_error = second_error
                break
    raise DataLoadError(
        "Could not read this CSV file. It may be corrupted or badly formatted. "
        f"(Technical detail: {str(last_error)[:150]})"
    )


def _read_excel(file_bytes: bytes, ext: str, sheet_name) -> pd.DataFrame:
    try:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name if sheet_name is not None else 0,
            engine=_excel_engine(ext),
        )
    except ImportError:
        raise DataLoadError("A required library for this Excel format is missing. Run: pip install -r requirements.txt")
    except Exception as error:
        raise DataLoadError(
            "Could not read this Excel sheet. The file may be corrupted or password-protected. "
            f"(Technical detail: {str(error)[:150]})"
        )


def load_dataset(file_bytes: bytes, filename: str, sheet_name=None) -> pd.DataFrame:
    """Validate and read an uploaded file into a DataFrame. The file is only read, never executed."""
    ext = validate_file(filename, len(file_bytes))
    df = _read_csv(file_bytes) if ext == ".csv" else _read_excel(file_bytes, ext, sheet_name)
    df.columns = [str(c) for c in df.columns]

    if df.shape[1] == 0:
        raise DataLoadError("No columns were found in this file.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has column headings but no data rows.")
    return df


# --------------------------------------------------------------------------
# Converting text columns to numbers / dates (used for detection and checks)
# --------------------------------------------------------------------------
def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert values to numbers. Anything that cannot be converted becomes NaN."""
    if ptypes.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    cleaned = series.astype("string").str.replace(CURRENCY_PATTERN, "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def to_datetime_series(series: pd.Series) -> pd.Series:
    """Convert values to dates. Anything that cannot be converted becomes NaT."""
    if ptypes.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


def _date_success_rate(texts: pd.Series) -> float:
    sample = texts.head(500)
    if sample.empty:
        return 0.0
    looks_like_date = sample.str.contains(DATE_PATTERN, regex=True)
    if looks_like_date.mean() < 0.8:
        return 0.0
    parsed = to_datetime_series(sample)
    return float((parsed.notna() & looks_like_date).mean())


# --------------------------------------------------------------------------
# Column type detection
# --------------------------------------------------------------------------
def _has_id_name(name: str) -> bool:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    tokens = re.sub(r"[^a-z0-9]+", " ", spaced.lower()).split()
    compact = "".join(tokens)
    if not tokens:
        return False
    return (
        tokens[-1] in ID_TOKENS
        or tokens[0] in ID_FIRST_TOKENS
        or compact.endswith(ID_COMPACT_SUFFIXES)
    )


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """True where a cell is empty (NaN) or contains only spaces."""
    mask = df.isna()
    for col in df.columns:
        s = df[col]
        if not (ptypes.is_numeric_dtype(s) or ptypes.is_datetime64_any_dtype(s) or ptypes.is_bool_dtype(s)):
            blank = s.astype("string").str.strip().eq("").fillna(False).astype(bool)
            mask[col] = mask[col] | blank
    return mask


def _classify_series(name: str, s: pd.Series) -> str:
    nn = s.dropna()
    if nn.empty:
        return "empty"
    if ptypes.is_bool_dtype(s):
        return "categorical"
    if ptypes.is_datetime64_any_dtype(s):
        return "date"

    n = len(nn)
    unique_ratio = nn.nunique() / n

    if ptypes.is_numeric_dtype(s):
        if _has_id_name(name):
            return "identifier"
        if ptypes.is_integer_dtype(s) and n >= 10 and unique_ratio == 1 and nn.is_monotonic_increasing:
            return "identifier"  # looks like a row counter 1, 2, 3...
        return "numeric"

    # Text columns: could still be numbers, dates, identifiers, or categories
    if _has_id_name(name):
        return "identifier"
    texts = nn.astype(str).str.strip()
    if to_numeric_series(texts).notna().mean() >= 0.8:
        return "numeric"
    if _date_success_rate(texts) >= 0.8:
        return "date"
    avg_len = texts.str.len().mean()
    if n >= 20 and unique_ratio >= 0.98 and avg_len <= 30:
        return "identifier"
    if texts.str.len().median() > 50:
        return "text"
    return "categorical"


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """Returns {column name: 'numeric' | 'categorical' | 'date' | 'identifier' | 'text' | 'empty'}."""
    clean = df.mask(missing_mask(df))
    return {col: _classify_series(col, clean[col]) for col in df.columns}


# --------------------------------------------------------------------------
# Data-quality report
# --------------------------------------------------------------------------
def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return -1


def quality_report(df: pd.DataFrame, col_types: dict[str, str]) -> dict:
    mask = missing_mask(df)
    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = int(mask.values.sum())

    profile = pd.DataFrame({
        "Column": list(df.columns),
        "Detected type": [col_types[c] for c in df.columns],
        "Stored as": [str(df[c].dtype) for c in df.columns],
        "Non-missing": (rows - mask.sum()).values,
        "Missing": mask.sum().values,
        "Missing %": (mask.mean() * 100).round(1).values,
        "Unique values": [_safe_nunique(df[c]) for c in df.columns],
    })

    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:
        duplicate_rows = 0

    empty_columns = [c for c, t in col_types.items() if t == "empty"]
    constant_columns = [
        c for c in df.columns
        if col_types[c] != "empty" and rows > 1 and _safe_nunique(df[c]) == 1
    ]

    # Values that do not fit the column's detected type (e.g. the word "abc" in a numeric column)
    clean = df.mask(mask)
    invalid_rows = []
    for col, kind in col_types.items():
        if kind not in ("numeric", "date"):
            continue
        if ptypes.is_numeric_dtype(df[col]) or ptypes.is_datetime64_any_dtype(df[col]):
            continue
        s = clean[col].dropna()
        if s.empty:
            continue
        texts = s.astype(str).str.strip()
        parsed = to_numeric_series(texts) if kind == "numeric" else to_datetime_series(texts)
        bad = texts[parsed.isna()]
        if len(bad) > 0:
            invalid_rows.append({
                "Column": col,
                "Expected type": kind,
                "Invalid values": int(len(bad)),
                "Examples": ", ".join(map(str, bad.unique()[:3])),
            })

    return {
        "rows": rows,
        "columns": cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": (missing_cells / total_cells * 100) if total_cells else 0.0,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": (duplicate_rows / rows * 100) if rows else 0.0,
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "invalid_values": pd.DataFrame(invalid_rows),
        "column_profile": profile,
        "missing_by_column": profile[["Column", "Missing", "Missing %"]]
        .sort_values("Missing", ascending=False)
        .reset_index(drop=True),
    }


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame, col_types: dict[str, str]) -> pd.DataFrame:
    cols = [c for c, t in col_types.items() if t == "numeric"]
    if not cols:
        return pd.DataFrame()
    numeric_df = pd.DataFrame({c: to_numeric_series(df[c]) for c in cols})
    return numeric_df.describe().T.round(2)


def categorical_summary(df: pd.DataFrame, col_types: dict[str, str]) -> pd.DataFrame:
    clean = df.mask(missing_mask(df))
    rows = []
    for col, kind in col_types.items():
        if kind not in ("categorical", "identifier", "text"):
            continue
        s = clean[col].dropna()
        if s.empty:
            continue
        counts = s.astype(str).value_counts()
        rows.append({
            "Column": col,
            "Type": kind,
            "Unique values": int(s.nunique()),
            "Most common": counts.index[0],
            "Times seen": int(counts.iloc[0]),
        })
    return pd.DataFrame(rows)
