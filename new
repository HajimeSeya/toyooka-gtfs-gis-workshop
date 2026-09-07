"""GTFS×250m将来人口演習の共通処理。

豊岡市コバス版と任意市区町村版から利用します。講習では原則として
このファイルを直接編集する必要はありません。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import os
import re
import shutil
import zipfile

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
import requests
from branca.colormap import linear
from folium.plugins import TimestampedGeoJson
from IPython.display import display
from shapely.geometry import LineString


@dataclass
class Workshop:
    gtfs: dict[str, pd.DataFrame]
    population: gpd.GeoDataFrame
    data_dir: Path
    output_dir: Path


@dataclass
class ServiceResult:
    date: pd.Timestamp
    active_services: set[str]
    active_trips: pd.DataFrame
    active_stop_times: pd.DataFrame
    trip_summary: pd.DataFrame
    route_summary: pd.DataFrame
    route_lines: gpd.GeoDataFrame
    output_dir: Path


@dataclass
class AccessResult:
    distance_m: int
    population_year: int
    metric_crs: object
    population_metric: gpd.GeoDataFrame
    catchment_union: object
    coverage_summary: pd.DataFrame
    stop_access: gpd.GeoDataFrame
    output_dir: Path


def _download(url: str | Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    local_source = Path(str(url)).expanduser()
    if local_source.exists():
        if local_source.resolve() != destination.resolve():
            shutil.copy2(local_source, destination)
        return destination
    test_dir = os.environ.get("WORKSHOP_TEST_DATA_DIR")
    if test_dir:
        local = Path(test_dir) / destination.name
        if local.exists():
            shutil.copy2(local, destination)
            return destination
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _read_gtfs(path: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                tables[Path(name).stem] = pd.read_csv(
                    io.BytesIO(zf.read(name)), dtype=str, encoding="utf-8-sig"
                )
    return tables


def load_workshop(
    gtfs_url: str | Path,
    population_url: str | Path,
    data_dir: str | Path = "toyooka_workshop",
    output_dir: str | Path = "output",
) -> Workshop:
    """GTFSと人口データを取得し、分析できる状態にする。"""
    if "REPLACE_WITH" in str(population_url):
        raise ValueError("講師が人口データのGitHub Release URLを設定してください。")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gtfs_zip = _download(gtfs_url, data_dir / "gtfs.zip")
    population_zip = _download(
        population_url,
        data_dir / "population_250m_workshop.geojson.zip",
    )

    population_dir = data_dir / "population"
    if population_dir.exists():
        shutil.rmtree(population_dir)
    population_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(population_zip) as zf:
        zf.extractall(population_dir)

    gtfs = _read_gtfs(gtfs_zip)
    routes = gtfs["routes"]
    if "route_long_name" not in routes.columns:
        routes["route_long_name"] = routes.get("route_short_name", routes["route_id"])
    routes["route_long_name"] = routes["route_long_name"].fillna(
        routes.get("route_short_name", routes["route_id"])
    )
    if "route_color" not in routes.columns:
        routes["route_color"] = "3388cc"
    routes["route_color"] = routes["route_color"].fillna("3388cc")
    population_file = next(population_dir.glob("*.geojson"))
    population = gpd.read_file(population_file)
    required_population_columns = {"pop_2020", "pop_2025", "age65p_2025"}
    missing_population_columns = required_population_columns - set(population.columns)
    if missing_population_columns:
        if "PTN_2020" in population.columns:
            raise ValueError(
                "兵庫県全域の原データではなく、講習用に加工した "
                "市区町村別に加工した workshop.geojson.zip を指定してください。"
            )
        raise ValueError(
            "人口データに必要な列がありません："
            + "、".join(sorted(missing_population_columns))
        )

    stops = gtfs["stops"]
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"])
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"])
    stop_times = gtfs["stop_times"]
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"])
    stop_times["departure_sec"] = stop_times["departure_time"].map(_gtfs_seconds)
    stop_times["arrival_sec"] = stop_times["arrival_time"].map(_gtfs_seconds)

    print("データの準備ができました。")
    print(
        f"路線 {len(gtfs['routes'])}、停留所レコード {len(stops)}、"
        f"人口メッシュ {len(population):,}個"
    )
    return Workshop(
        gtfs=gtfs,
        population=population,
        data_dir=data_dir,
        output_dir=output_dir,
    )


def available_population_years(workshop: Workshop) -> list[int]:
    """総人口と65歳以上人口の両方を分析できる将来年を返す。"""
    population = workshop.population
    years = []
    for column in population.columns:
        match = re.fullmatch(r"pop_(\d{4})", str(column))
        if match and f"age65p_{match.group(1)}" in population.columns:
            years.append(int(match.group(1)))
    return sorted(set(years))


def _save_table(table: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _save_figure(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return path


def _save_map(map_object, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    map_object.save(path)
    return path


def show_gtfs_tables(workshop: Workshop) -> None:
    """GTFSのファイル構成と主要2表を表示する。"""
    summary = pd.DataFrame(
        {
            "ファイル": [f"{name}.txt" for name in workshop.gtfs],
            "行数": [len(table) for table in workshop.gtfs.values()],
        }
    )
    display(summary)
    _save_table(summary, workshop.output_dir, "gtfs_file_summary.csv")
    display(
        workshop.gtfs["routes"][["route_id", "route_long_name"]].rename(
            columns={"route_id": "路線ID", "route_long_name": "路線名"}
        )
    )
    display(
        workshop.gtfs["stops"][["stop_id", "stop_name", "stop_lat", "stop_lon"]]
        .head(10)
        .rename(
            columns={
                "stop_id": "停留所ID",
                "stop_name": "停留所名",
                "stop_lat": "緯度",
                "stop_lon": "経度",
            }
        )
    )


def check_gtfs(workshop: Workshop) -> pd.DataFrame:
    """講習用の簡易GTFS整合性チェックを行う。"""
    gtfs = workshop.gtfs
    required = {"agency", "routes", "trips", "stops", "stop_times"}
    missing = required - set(gtfs)
    stop_times = gtfs["stop_times"]
    stops = gtfs["stops"]
    trips = gtfs["trips"]
    orphan_stops = set(stop_times["stop_id"]) - set(stops["stop_id"])
    orphan_trips = set(stop_times["trip_id"]) - set(trips["trip_id"])
    result = pd.DataFrame(
        [
            ["必須ファイル", "OK" if not missing else "要確認", ", ".join(missing) or "不足なし"],
            ["停留所ID", "OK" if not orphan_stops else "要確認", f"未対応 {len(orphan_stops)}件"],
            ["便ID", "OK" if not orphan_trips else "要確認", f"未対応 {len(orphan_trips)}件"],
            ["路線形状", "OK" if "shapes" in gtfs else "注意", "shapes.txtなし：概略線を使用"],
        ],
        columns=["確認項目", "結果", "内容"],
    )
    display(result)
    _save_table(result, workshop.output_dir, "gtfs_check.csv")
    return result


def show_population_summary(workshop: Workshop) -> pd.DataFrame:
    """収録年の人口推移を表と折れ線で示し、CSVとPNGを保存する。"""
    population = workshop.population
    years = sorted(
        int(match.group(1))
        for column in population.columns
        if (match := re.fullmatch(r"pop_(\d{4})", str(column)))
    )
    rows = []
    for year in years:
        row = {"年": year, "総人口": round(population[f"pop_{year}"].fillna(0).sum())}
        older = f"age65p_{year}"
        row["65歳以上人口"] = (
            round(population[older].fillna(0).sum()) if older in population.columns else np.nan
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    display(summary)
    _save_table(summary, workshop.output_dir, "population_by_year.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(summary["年"], summary["総人口"], marker="o", linewidth=2.2, label="総人口")
    if summary["65歳以上人口"].notna().any():
        ax.plot(
            summary["年"],
            summary["65歳以上人口"],
            marker="o",
            linewidth=2.0,
            label="65歳以上人口",
        )
    ax.set_ylabel("人口（人）")
    ax.set_title("人口の推移（2025年以降は推計）")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, workshop.output_dir, "population_by_year.png")
    plt.show()
    return summary


def _gtfs_seconds(value: str) -> float:
    try:
        hour, minute, second = map(int, value.split(":"))
        return hour * 3600 + minute * 60 + second
    except (AttributeError, TypeError, ValueError):
        return np.nan


def _active_service_ids(
    date: pd.Timestamp,
    calendar: pd.DataFrame | None,
    exceptions: pd.DataFrame | None,
) -> set[str]:
    date = pd.Timestamp(date).normalize()
    active: set[str] = set()
    if calendar is not None and not calendar.empty:
        weekday = date.day_name().lower()
        start = pd.to_datetime(calendar["start_date"], format="%Y%m%d")
        end = pd.to_datetime(calendar["end_date"], format="%Y%m%d")
        mask = (start <= date) & (date <= end) & (calendar[weekday] == "1")
        active = set(calendar.loc[mask, "service_id"])
    if exceptions is not None and not exceptions.empty:
        exception_dates = pd.to_datetime(exceptions["date"], format="%Y%m%d")
        selected = exceptions.loc[exception_dates == date]
        active.update(selected.loc[selected["exception_type"] == "1", "service_id"])
        active.difference_update(selected.loc[selected["exception_type"] == "2", "service_id"])
    return active


def suggest_analysis_date(workshop: Workshop, preferred_weekday: int = 2) -> str:
    """GTFS有効期間から、運行のある代表日（既定は水曜日）を1日提案する。"""
    calendar = workshop.gtfs.get("calendar")
    exceptions = workshop.gtfs.get("calendar_dates")
    starts = []
    ends = []
    if calendar is not None and not calendar.empty:
        starts.append(pd.to_datetime(calendar["start_date"], format="%Y%m%d").min())
        ends.append(pd.to_datetime(calendar["end_date"], format="%Y%m%d").max())
    if exceptions is not None and not exceptions.empty:
        dates = pd.to_datetime(exceptions["date"], format="%Y%m%d")
        starts.append(dates.min())
        ends.append(dates.max())
    if not starts:
        raise ValueError("GTFSにサービス期間を判断できる情報がありません。")
    trips = workshop.gtfs["trips"]
    for require_weekday in (True, False):
        for date in pd.date_range(min(starts), max(ends)):
            if require_weekday and date.weekday() != preferred_weekday:
                continue
            services = _active_service_ids(date, calendar, exceptions)
            if trips["service_id"].isin(services).any():
                return str(date.date())
    raise ValueError("GTFS有効期間内に運行便を見つけられません。")


def select_service(workshop: Workshop, date: str) -> ServiceResult:
    """指定日に運行する便と路線別指標を作る。"""
    gtfs = workshop.gtfs
    analysis_date = pd.Timestamp(date)
    services = _active_service_ids(
        analysis_date, gtfs.get("calendar"), gtfs.get("calendar_dates")
    )
    trips = gtfs["trips"]
    routes = gtfs["routes"].copy()
    if "route_long_name" not in routes.columns:
        routes["route_long_name"] = routes.get("route_short_name", routes["route_id"])
    routes["route_long_name"] = routes["route_long_name"].fillna(
        routes.get("route_short_name", routes["route_id"])
    )
    if "route_color" not in routes.columns:
        routes["route_color"] = "3388cc"
    routes["route_color"] = routes["route_color"].fillna("3388cc")
    stop_times = gtfs["stop_times"]
    stops = gtfs["stops"]
    active_trips = trips.loc[trips["service_id"].isin(services)].copy()
    if active_trips.empty:
        raise ValueError("この日は運行便がありません。分析日を確認してください。")
    active_stop_times = stop_times.merge(
        active_trips[["trip_id", "route_id"]], on="trip_id", how="inner"
    )
    first = active_stop_times.loc[
        active_stop_times.groupby("trip_id")["stop_sequence"].idxmin()
    ]
    last = active_stop_times.loc[
        active_stop_times.groupby("trip_id")["stop_sequence"].idxmax()
    ]
    if "trip_headsign" not in active_trips.columns:
        active_trips["trip_headsign"] = ""
    trip_summary = active_trips[["trip_id", "route_id", "trip_headsign"]].merge(
        first[["trip_id", "departure_time", "departure_sec"]], on="trip_id"
    ).merge(last[["trip_id", "arrival_time", "arrival_sec"]], on="trip_id")

    rows = []
    for route_id, group in trip_summary.groupby("route_id"):
        departures = np.sort(group["departure_sec"].dropna().to_numpy())
        headways = np.diff(departures) / 60
        rows.append(
            {
                "route_id": route_id,
                "便数": len(group),
                "始発": group.loc[group["departure_sec"].idxmin(), "departure_time"][:5],
                "最終便の終着": group.loc[group["arrival_sec"].idxmax(), "arrival_time"][:5],
                "運行間隔中央値（分）": round(float(np.median(headways)), 1)
                if len(headways)
                else np.nan,
            }
        )
    route_summary = routes[["route_id", "route_long_name", "route_color"]].merge(
        pd.DataFrame(rows), on="route_id", how="left"
    ).rename(columns={"route_long_name": "路線名"})

    line_rows = []
    shapes = gtfs.get("shapes")
    if shapes is not None and "shape_id" in active_trips.columns:
        shapes = shapes.copy()
        for column in ["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]:
            shapes[column] = pd.to_numeric(shapes[column], errors="coerce")
        route_shapes = active_trips[["route_id", "shape_id"]].dropna().drop_duplicates()
        for item in route_shapes.itertuples(index=False):
            ordered = shapes.loc[shapes["shape_id"] == item.shape_id].sort_values("shape_pt_sequence")
            ordered = ordered.dropna(subset=["shape_pt_lon", "shape_pt_lat"])
            if len(ordered) >= 2:
                line_rows.append(
                    {
                        "route_id": item.route_id,
                        "line_kind": "shapes.txtの路線形状",
                        "geometry": LineString(zip(ordered["shape_pt_lon"], ordered["shape_pt_lat"])),
                    }
                )
    if not line_rows:
        for route_id, group in active_trips.groupby("route_id"):
            patterns: set[tuple[str, ...]] = set()
            for trip_id in group["trip_id"]:
                ordered_times = stop_times.loc[stop_times["trip_id"] == trip_id].sort_values("stop_sequence")
                pattern = tuple(ordered_times["stop_id"].astype(str))
                if len(pattern) < 2 or pattern in patterns:
                    continue
                patterns.add(pattern)
                ordered = ordered_times.merge(
                    stops[["stop_id", "stop_lon", "stop_lat"]], on="stop_id"
                ).dropna(subset=["stop_lon", "stop_lat"])
                if len(ordered) >= 2:
                    line_rows.append(
                        {
                            "route_id": route_id,
                            "line_kind": "停留所を結んだ概略線",
                            "geometry": LineString(zip(ordered["stop_lon"], ordered["stop_lat"])),
                        }
                    )
    route_lines = gpd.GeoDataFrame(line_rows, crs=4326).merge(
        routes[["route_id", "route_long_name", "route_color"]], on="route_id"
    )
    weekday = ["月", "火", "水", "木", "金", "土", "日"][analysis_date.weekday()]
    print(f"{analysis_date.date()}（{weekday}曜日）：{len(active_trips)}便")
    return ServiceResult(
        date=analysis_date,
        active_services=services,
        active_trips=active_trips,
        active_stop_times=active_stop_times,
        trip_summary=trip_summary,
        route_summary=route_summary,
        route_lines=route_lines,
        output_dir=workshop.output_dir,
    )


def show_route_summary(service: ServiceResult) -> None:
    table = service.route_summary[
        ["路線名", "便数", "始発", "最終便の終着", "運行間隔中央値（分）"]
    ]
    display(table)
    _save_table(table, service.output_dir, f"route_summary_{service.date.date()}.csv")


def show_service_timeline(service: ServiceResult) -> None:
    route_order = service.route_summary["route_id"].tolist()
    names = service.route_summary.set_index("route_id")["路線名"].to_dict()
    colors = {
        row.route_id: "#" + (row.route_color if pd.notna(row.route_color) else "3388cc")
        for row in service.route_summary.itertuples()
    }
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for y, route_id in enumerate(route_order):
        group = service.trip_summary.loc[service.trip_summary["route_id"] == route_id]
        ax.scatter(
            group["departure_sec"] / 3600,
            [y] * len(group),
            s=90,
            color=colors[route_id],
            edgecolor="white",
        )
    ax.set_yticks(range(len(route_order)), [names[r] for r in route_order])
    ax.set_xlim(7, 20)
    ax.set_xticks(range(7, 21), [f"{h}:00" for h in range(7, 21)])
    ax.set_xlabel("出発時刻")
    ax.set_title(f"コバスの運行（{service.date.date()}）")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    _save_figure(fig, service.output_dir, f"service_timeline_{service.date.date()}.png")
    plt.show()


def _route_id_from_name(service: ServiceResult, route: str) -> str:
    """路線名またはroute_idをroute_idへ変換する。"""
    routes = service.route_summary
    route = str(route)
    by_id = routes.loc[routes["route_id"].astype(str) == route]
    by_name = routes.loc[routes["路線名"] == route]
    selected = by_id if len(by_id) else by_name
    if selected.empty:
        choices = "、".join(routes["路線名"].dropna().astype(str))
        raise ValueError(f"路線が見つかりません。次から選んでください：{choices}")
    return str(selected.iloc[0]["route_id"])


def _named_active_stop_times(
    workshop: Workshop, service: ServiceResult
) -> pd.DataFrame:
    """指定日のstop_timesに停留所名・座標・路線名を付ける。"""
    return (
        service.active_stop_times.merge(
            workshop.gtfs["stops"][
                ["stop_id", "stop_name", "stop_lat", "stop_lon"]
            ],
            on="stop_id",
            how="left",
        )
        .merge(
            workshop.gtfs["routes"][
                ["route_id", "route_long_name", "route_color"]
            ],
            on="route_id",
            how="left",
        )
    )


def show_stop_hour_heatmap(
    workshop: Workshop,
    service: ServiceResult,
    stop_names: list[str] | None = None,
) -> pd.DataFrame:
    """主要停留所について、時間帯別の停車回数をヒートマップで示す。"""
    active = _named_active_stop_times(workshop, service).dropna(
        subset=["arrival_sec", "stop_name"]
    )
    active["hour"] = (active["arrival_sec"] // 3600).astype(int)

    if stop_names is None:
        candidates = [
            "豊岡駅",
            "市役所前",
            "芸術文化観光専門職大学",
            "総合健康ゾーン（ウェルストーク豊岡）",
            "コープデイズ前",
            "長寿園前",
            "バザールタウン豊岡メガ・フレッシュ館前",
            "フレッシュバザール豊岡九日市店前",
        ]
        available = set(active["stop_name"])
        stop_names = [name for name in candidates if name in available]
        busiest = active["stop_name"].value_counts().index.tolist()
        stop_names += [name for name in busiest if name not in stop_names][
            : max(0, 10 - len(stop_names))
        ]
    else:
        stop_names = [name for name in stop_names if name in set(active["stop_name"])]

    if not stop_names:
        raise ValueError("表示できる停留所がありません。停留所名を確認してください。")

    first_hour = int(active["hour"].min())
    last_hour = int(active["hour"].max())
    hours = list(range(first_hour, last_hour + 1))
    table = pd.crosstab(active["stop_name"], active["hour"]).reindex(
        index=stop_names, columns=hours, fill_value=0
    )

    fig_height = max(4.8, 0.52 * len(stop_names))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    image = ax.imshow(table.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(hours)), [f"{hour}時" for hour in hours])
    ax.set_yticks(range(len(stop_names)), stop_names)
    ax.set_xlabel("時間帯")
    ax.set_title(f"主要停留所の時間帯別停車回数（{service.date.date()}）")
    for row in range(len(stop_names)):
        for col in range(len(hours)):
            value = int(table.iloc[row, col])
            if value:
                ax.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= max(2, table.to_numpy().max() * 0.55) else "#222222",
                )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("停車回数")
    plt.tight_layout()
    _save_table(
        table.reset_index().rename(columns={"stop_name": "停留所名"}),
        service.output_dir,
        f"stop_hour_heatmap_{service.date.date()}.csv",
    )
    _save_figure(fig, service.output_dir, f"stop_hour_heatmap_{service.date.date()}.png")
    plt.show()
    return table


def show_time_space_diagram(
    workshop: Workshop,
    service: ServiceResult,
    route: str = "コバス北ルート",
) -> None:
    """横軸を時刻、縦軸を停留所順序とする運行図を描く。"""
    route_id = _route_id_from_name(service, route)
    route_row = service.route_summary.loc[
        service.route_summary["route_id"].astype(str) == route_id
    ].iloc[0]
    route_name = route_row["路線名"]
    color = "#" + (
        route_row["route_color"] if pd.notna(route_row["route_color"]) else "3388cc"
    )
    active = _named_active_stop_times(workshop, service)
    active = active.loc[active["route_id"].astype(str) == route_id].copy()
    if active.empty:
        raise ValueError(f"{route_name}は指定日に運行していません。")

    trip_sizes = active.groupby("trip_id").size()
    representative_id = trip_sizes.idxmax()
    representative = active.loc[active["trip_id"] == representative_id].sort_values(
        "stop_sequence"
    )
    max_sequence = int(representative["stop_sequence"].max())

    fig_height = max(6.2, min(10.0, max_sequence * 0.24))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for _, group in active.groupby("trip_id"):
        ordered = group.sort_values("stop_sequence").dropna(subset=["arrival_sec"])
        ax.plot(
            ordered["arrival_sec"] / 3600,
            ordered["stop_sequence"],
            color=color,
            linewidth=1.8,
            alpha=0.78,
        )
        first = ordered.iloc[0]
        ax.text(
            first["arrival_sec"] / 3600,
            first["stop_sequence"] - 0.35,
            first["arrival_time"][:5],
            fontsize=7,
            color=color,
            ha="center",
        )

    key_stops = {
        "豊岡駅",
        "市役所前",
        "芸術文化観光専門職大学",
        "総合健康ゾーン（ウェルストーク豊岡）",
    }
    tick_rows = representative.loc[
        (representative["stop_sequence"] == 1)
        | (representative["stop_sequence"] == max_sequence)
        | (representative["stop_sequence"] % 3 == 0)
        | representative["stop_name"].isin(key_stops)
    ].drop_duplicates("stop_sequence")
    ax.set_yticks(tick_rows["stop_sequence"], tick_rows["stop_name"])
    start_hour = int(np.floor(active["arrival_sec"].min() / 3600))
    end_hour = int(np.ceil(active["arrival_sec"].max() / 3600))
    ax.set_xticks(range(start_hour, end_hour + 1), [f"{h}:00" for h in range(start_hour, end_hour + 1)])
    ax.set_xlim(start_hour - 0.05, end_hour + 0.05)
    ax.set_ylim(0.4, max_sequence + 0.6)
    ax.set_xlabel("時刻")
    ax.set_ylabel("停留所の順序")
    ax.set_title(f"{route_name}の時空間ダイヤ（{service.date.date()}）")
    ax.grid(alpha=0.22)
    plt.tight_layout()
    safe_route = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠ー_-]+", "_", str(route_name))
    _save_figure(
        fig,
        service.output_dir,
        f"time_space_{safe_route}_{service.date.date()}.png",
    )
    plt.show()


def _base_map(center):
    result = folium.Map(location=center, zoom_start=14, tiles=None)
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        attr="国土地理院 淡色地図",
        name="国土地理院 淡色地図",
    ).add_to(result)
    return result


def _add_routes(target, route_lines, layer_name="路線"):
    """各路線を独立したレイヤーとして追加し、チェック切替を可能にする。"""
    for route_id, group in route_lines.groupby("route_id", sort=False):
        first = group.iloc[0]
        route_name = str(first.get("route_long_name", route_id))
        color_value = first.get("route_color", "3388cc")
        color = "#" + (color_value if pd.notna(color_value) and color_value else "3388cc")
        layer = folium.FeatureGroup(name=f"{layer_name}：{route_name}", show=True)
        for row in group.itertuples():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            line_kind = getattr(row, "line_kind", "路線形状")
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                dash_array="7,5",
                tooltip=f"{route_name}（{line_kind}）",
            ).add_to(layer)
        layer.add_to(target)


def make_gtfs_map(workshop: Workshop, service: ServiceResult):
    stops = workshop.gtfs["stops"]
    center = [stops["stop_lat"].mean(), stops["stop_lon"].mean()]
    result = _base_map(center)
    _add_routes(result, service.route_lines)
    stop_layer = folium.FeatureGroup(name="停留所", show=True)
    for row in stops.itertuples():
        folium.CircleMarker(
            [row.stop_lat, row.stop_lon],
            radius=4,
            color="#222222",
            fill=True,
            fill_color="white",
            fill_opacity=1,
            tooltip=row.stop_name,
        ).add_to(stop_layer)
    stop_layer.add_to(result)
    folium.LayerControl(collapsed=False).add_to(result)
    _save_map(result, workshop.output_dir, f"gtfs_routes_{service.date.date()}.html")
    return result


def make_reachable_map(
    workshop: Workshop,
    service: ServiceResult,
    origin: str = "豊岡駅",
    departure: str = "09:00",
    max_minutes: int = 60,
):
    """指定時刻以降に乗車し、乗換なしで到達できる停留所を示す。"""
    try:
        hour, minute = map(int, departure.split(":")[:2])
        start_sec = hour * 3600 + minute * 60
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("出発時刻は '09:00' のように入力してください。") from exc

    active = _named_active_stop_times(workshop, service).dropna(
        subset=["arrival_sec", "departure_sec", "stop_name"]
    )
    origins = active.loc[
        (active["stop_name"] == origin) & (active["departure_sec"] >= start_sec)
    ].sort_values("departure_sec")
    if origins.empty:
        choices = "、".join(sorted(active["stop_name"].dropna().unique()))
        raise ValueError(
            f"{departure}以降に「{origin}」を出発する便がありません。"
            f"停留所名の候補：{choices}"
        )

    limit_sec = start_sec + int(max_minutes) * 60
    rows = []
    for origin_row in origins.itertuples():
        downstream = active.loc[
            (active["trip_id"] == origin_row.trip_id)
            & (active["stop_sequence"] >= origin_row.stop_sequence)
            & (active["arrival_sec"] <= limit_sec)
        ]
        for row in downstream.itertuples():
            rows.append(
                {
                    "stop_name": row.stop_name,
                    "arrival_sec": row.arrival_sec,
                    "route_long_name": row.route_long_name,
                    "stop_lat": row.stop_lat,
                    "stop_lon": row.stop_lon,
                }
            )
    if not rows:
        raise ValueError(f"{max_minutes}分以内に到達できる停留所がありません。")

    reachable = (
        pd.DataFrame(rows)
        .sort_values("arrival_sec")
        .drop_duplicates("stop_name")
        .copy()
    )
    reachable["経過分"] = ((reachable["arrival_sec"] - start_sec) / 60).round().astype(int)
    reachable["到着時刻"] = reachable["arrival_sec"].map(
        lambda sec: f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}"
    )

    stops = workshop.gtfs["stops"]
    origin_rows = stops.loc[stops["stop_name"] == origin]
    center = (
        [origin_rows["stop_lat"].mean(), origin_rows["stop_lon"].mean()]
        if len(origin_rows)
        else [stops["stop_lat"].mean(), stops["stop_lon"].mean()]
    )
    result = _base_map(center)
    _add_routes(result, service.route_lines)
    color_map = linear.YlGnBu_09.scale(0, max_minutes)
    color_map.caption = f"{departure}からの経過時間（分、乗換なし）"
    for row in reachable.itertuples():
        color = color_map(min(max_minutes, row.経過分))
        folium.CircleMarker(
            [row.stop_lat, row.stop_lon],
            radius=7,
            color="#333333",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"{row.stop_name}：{row.到着時刻}",
            popup=folium.Popup(
                f"<b>{row.stop_name}</b><br>最早到着：{row.到着時刻}<br>"
                f"出発時刻から：{row.経過分}分<br>路線：{row.route_long_name}",
                max_width=320,
            ),
        ).add_to(result)
    if len(origin_rows):
        folium.Marker(
            center,
            tooltip=f"出発地：{origin}",
            popup=f"{origin}を{departure}に出発すると仮定",
            icon=folium.Icon(color="red", icon="play"),
        ).add_to(result)
    color_map.add_to(result)
    folium.LayerControl(collapsed=False).add_to(result)
    print(
        f"{origin}を{departure}に出発：乗換なし・{max_minutes}分以内に"
        f"{len(reachable)}停留所へ到達可能"
    )
    _save_map(
        result,
        workshop.output_dir,
        f"reachable_{service.date.date()}_{departure.replace(':', '')}_{max_minutes}min.html",
    )
    return result


def make_schedule_animation(
    workshop: Workshop,
    service: ServiceResult,
    step_minutes: int = 2,
):
    """GTFS時刻表から推定したバスの予定位置を時間スライダーで示す。"""
    if step_minutes < 1:
        raise ValueError("step_minutesは1以上にしてください。")
    active = _named_active_stop_times(workshop, service).dropna(
        subset=["arrival_sec", "stop_lat", "stop_lon"]
    )
    features = []
    step_seconds = int(step_minutes) * 60
    for trip_id, group in active.groupby("trip_id"):
        ordered = (
            group.sort_values(["arrival_sec", "stop_sequence"])
            .drop_duplicates("arrival_sec", keep="last")
        )
        if len(ordered) < 2:
            continue
        seconds = ordered["arrival_sec"].to_numpy(dtype=float)
        samples = np.arange(
            np.ceil(seconds.min() / step_seconds) * step_seconds,
            seconds.max() + 1,
            step_seconds,
        )
        samples = np.unique(np.r_[seconds.min(), samples, seconds.max()])
        lons = np.interp(samples, seconds, ordered["stop_lon"].to_numpy(dtype=float))
        lats = np.interp(samples, seconds, ordered["stop_lat"].to_numpy(dtype=float))
        route_name = str(ordered.iloc[0]["route_long_name"])
        route_color = ordered.iloc[0]["route_color"]
        color = "#" + (route_color if pd.notna(route_color) else "3388cc")
        for second, lon, lat in zip(samples, lons, lats):
            timestamp = service.date.normalize() + pd.to_timedelta(second, unit="s")
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "time": timestamp.isoformat(),
                        "popup": f"{route_name}<br>{timestamp.strftime('%H:%M')}",
                        "tooltip": route_name,
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.95,
                            "stroke": True,
                            "radius": 6,
                            "color": "#222222",
                            "weight": 1,
                        },
                    },
                }
            )

    stops = workshop.gtfs["stops"]
    result = _base_map([stops["stop_lat"].mean(), stops["stop_lon"].mean()])
    _add_routes(result, service.route_lines)
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=f"PT{step_minutes}M",
        duration=f"PT{step_minutes}M",
        add_last_point=False,
        auto_play=False,
        loop=False,
        loop_button=True,
        max_speed=8,
        date_options="HH:mm",
        time_slider_drag_update=True,
    ).add_to(result)
    warning = (
        "<div style='position:fixed;top:10px;left:50px;z-index:9999;"
        "background:white;padding:7px 10px;border:1px solid #777;font-size:12px'>"
        "GTFS時刻表から推定した予定位置（実車位置ではありません）</div>"
    )
    result.get_root().html.add_child(folium.Element(warning))
    folium.LayerControl(collapsed=False).add_to(result)
    _save_map(result, workshop.output_dir, f"schedule_animation_{service.date.date()}.html")
    return result


def analyze_access(
    workshop: Workshop,
    service: ServiceResult,
    distance_m: int = 300,
    year: int = 2025,
) -> AccessResult:
    """停留所圏域の人口カバー率と停留所別近隣人口を計算する。"""
    year = int(year)
    available = available_population_years(workshop)
    if year not in available:
        raise ValueError(
            f"{year}年は分析できません。選択可能年：{', '.join(map(str, available))}"
        )
    gtfs = workshop.gtfs
    stops = gtfs["stops"]
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs=4326,
    )
    metric_crs = stops_gdf.estimate_utm_crs()
    stops_metric = stops_gdf.to_crs(metric_crs)
    population_metric = workshop.population.to_crs(metric_crs).copy()
    population_metric["mesh_area"] = population_metric.geometry.area
    catchment_union = stops_metric.geometry.buffer(distance_m).union_all()
    area = population_metric.geometry.intersection(catchment_union).area
    population_metric["covered_fraction"] = (
        area / population_metric["mesh_area"]
    ).clip(0, 1)

    coverage_rows = []
    for label, column in (("総人口", f"pop_{year}"), ("65歳以上", f"age65p_{year}")):
        total = population_metric[column].fillna(0).sum()
        covered = (
            population_metric[column].fillna(0)
            * population_metric["covered_fraction"]
        ).sum()
        coverage_rows.append(
            {
                "年": year,
                "人口区分": label,
                "市全体人口": round(total),
                f"停留所{distance_m}m圏人口": round(covered),
                "カバー率（%）": round(100 * covered / total, 1) if total else np.nan,
            }
        )
    coverage_summary = pd.DataFrame(coverage_rows)

    active_named = service.active_stop_times.merge(
        stops[["stop_id", "stop_name"]], on="stop_id", how="left"
    )
    calls = active_named.groupby("stop_name").size()
    stop_rows = []
    for stop_name, group in stops_metric.groupby("stop_name"):
        buffer = group.geometry.buffer(distance_m).union_all()
        candidates = population_metric.loc[
            population_metric.geometry.intersects(buffer)
        ].copy()
        if len(candidates):
            fraction = (
                candidates.geometry.intersection(buffer).area / candidates["mesh_area"]
            ).clip(0, 1)
            nearby_pop = (candidates[f"pop_{year}"].fillna(0) * fraction).sum()
            nearby_65p = (candidates[f"age65p_{year}"].fillna(0) * fraction).sum()
        else:
            nearby_pop = nearby_65p = 0
        stop_rows.append(
            {
                "停留所名": stop_name,
                "停車回数": int(calls.get(stop_name, 0)),
                f"{distance_m}m圏人口": round(nearby_pop),
                f"{distance_m}m圏65歳以上人口": round(nearby_65p),
                "geometry": group.geometry.union_all().centroid,
            }
        )
    stop_access = gpd.GeoDataFrame(stop_rows, crs=metric_crs).to_crs(4326)
    stop_access["人口／停車回数"] = (
        stop_access[f"{distance_m}m圏人口"]
        / stop_access["停車回数"].replace(0, np.nan)
    ).round(1)
    return AccessResult(
        distance_m=distance_m,
        population_year=year,
        metric_crs=metric_crs,
        population_metric=population_metric,
        catchment_union=catchment_union,
        coverage_summary=coverage_summary,
        stop_access=stop_access,
        output_dir=workshop.output_dir,
    )


def show_access_summary(access: AccessResult) -> None:
    display(access.coverage_summary)
    ranking = (
        access.stop_access.drop(columns="geometry")
        .sort_values("人口／停車回数", ascending=False)
        .head(12)
    )
    display(ranking)
    _save_table(
        access.coverage_summary,
        access.output_dir,
        f"coverage_{access.population_year}_{access.distance_m}m.csv",
    )
    _save_table(
        access.stop_access.drop(columns="geometry").sort_values(
            "人口／停車回数", ascending=False
        ),
        access.output_dir,
        f"stop_access_{access.population_year}_{access.distance_m}m.csv",
    )


def show_access_chart(access: AccessResult, service: ServiceResult) -> None:
    distance = access.distance_m
    table = access.stop_access
    x_col = "停車回数"
    y_col = f"{distance}m圏人口"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(table[x_col], table[y_col], s=65, alpha=0.75, color="#2b8cbe")
    for _, row in table.nlargest(6, "人口／停車回数").iterrows():
        ax.annotate(
            row["停留所名"],
            (row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel(f"停車回数（{service.date.date()}）")
    ax.set_ylabel(f"停留所{distance}m圏人口（{access.population_year}年推計）")
    ax.set_title(f"停車回数と近隣人口（{access.population_year}年）")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save_figure(
        fig,
        access.output_dir,
        f"calls_vs_population_{access.population_year}_{distance}m.png",
    )
    plt.show()


def make_access_map(
    workshop: Workshop,
    service: ServiceResult,
    access: AccessResult,
):
    distance = access.distance_m
    year = access.population_year
    stops = workshop.gtfs["stops"]
    center = [stops["stop_lat"].mean(), stops["stop_lon"].mean()]
    result = _base_map(center)
    population = workshop.population
    map_population = population.loc[
        population[f"pop_{year}"].fillna(0) > 0,
        ["mesh_id", f"pop_{year}", f"age65p_{year}", "geometry"],
    ].copy()
    vmax = max(1, float(map_population[f"pop_{year}"].quantile(0.98)))
    color_map = linear.YlOrRd_09.scale(0, vmax)
    color_map.caption = f"250mメッシュ人口（{year}年推計）"
    pop_layer = folium.FeatureGroup(name=f"250m人口（{year}年）", show=True)
    folium.GeoJson(
        map_population,
        style_function=lambda feature: {
            "fillColor": color_map(
                min(feature["properties"].get(f"pop_{year}") or 0, vmax)
            ),
            "color": "#777777",
            "weight": 0.25,
            "fillOpacity": 0.60,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[f"pop_{year}", f"age65p_{year}"],
            aliases=[f"{year}年人口", f"{year}年65歳以上人口"],
            localize=True,
        ),
    ).add_to(pop_layer)
    pop_layer.add_to(result)
    color_map.add_to(result)

    catchment = gpd.GeoSeries(
        [access.catchment_union], crs=access.metric_crs
    ).to_crs(4326)
    catchment_layer = folium.FeatureGroup(name=f"停留所{distance}m圏", show=True)
    folium.GeoJson(
        catchment,
        style_function=lambda feature: {
            "fillColor": "#3182bd",
            "color": "#08519c",
            "weight": 1,
            "fillOpacity": 0.14,
        },
    ).add_to(catchment_layer)
    catchment_layer.add_to(result)
    _add_routes(result, service.route_lines)

    stop_layer = folium.FeatureGroup(name="停留所", show=True)
    for _, row in access.stop_access.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=4 + min(8, row["停車回数"] / 3),
            color="#08306b",
            fill=True,
            fill_color="white",
            fill_opacity=0.95,
            tooltip=row["停留所名"],
            popup=folium.Popup(
                f"<b>{row['停留所名']}</b><br>"
                f"停車回数：{row['停車回数']}回<br>"
                f"{distance}m圏人口：{row[f'{distance}m圏人口']:,.0f}人<br>"
                f"うち65歳以上：{row[f'{distance}m圏65歳以上人口']:,.0f}人",
                max_width=300,
            ),
        ).add_to(stop_layer)
    stop_layer.add_to(result)
    folium.LayerControl(collapsed=False).add_to(result)
    _save_map(
        result,
        workshop.output_dir,
        f"access_map_{year}_{distance}m_{service.date.date()}.html",
    )
    return result


def make_service_review_map(
    workshop: Workshop,
    service: ServiceResult,
    access: AccessResult,
):
    """近隣人口と停車回数の組合せを4分類して地図に示す。"""
    distance = access.distance_m
    table = access.stop_access.copy()
    population_col = f"{distance}m圏人口"
    population_median = float(table[population_col].median())
    calls_median = float(table["停車回数"].median())

    def classify(row):
        high_population = row[population_col] >= population_median
        high_service = row["停車回数"] >= calls_median
        if high_population and not high_service:
            return "人口多・便少（要確認）", "#d73027"
        if high_population and high_service:
            return "人口多・便多", "#1a9850"
        if not high_population and high_service:
            return "人口少・便多", "#4575b4"
        return "人口少・便少", "#bdbdbd"

    classifications = table.apply(classify, axis=1)
    table["分類"] = [item[0] for item in classifications]
    table["色"] = [item[1] for item in classifications]

    stops = workshop.gtfs["stops"]
    result = _base_map([stops["stop_lat"].mean(), stops["stop_lon"].mean()])
    _add_routes(result, service.route_lines)
    for _, row in table.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=8,
            color="#333333",
            weight=1,
            fill=True,
            fill_color=row["色"],
            fill_opacity=0.9,
            tooltip=f"{row['停留所名']}：{row['分類']}",
            popup=folium.Popup(
                f"<b>{row['停留所名']}</b><br>{row['分類']}<br>"
                f"停車回数：{row['停車回数']}回<br>"
                f"{distance}m圏人口：{row[population_col]:,.0f}人",
                max_width=320,
            ),
        ).add_to(result)

    legend = """
    <div style="position:fixed;bottom:30px;left:40px;z-index:9999;
      background:white;padding:9px 12px;border:1px solid #777;font-size:12px">
      <b>近隣人口 × 停車回数</b><br>
      <span style="color:#d73027">●</span> 人口多・便少（要確認）<br>
      <span style="color:#1a9850">●</span> 人口多・便多<br>
      <span style="color:#4575b4">●</span> 人口少・便多<br>
      <span style="color:#bdbdbd">●</span> 人口少・便少<br>
      ※中央値による探索的な分類
    </div>
    """
    result.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(result)
    _save_map(
        result,
        workshop.output_dir,
        f"service_review_{access.population_year}_{distance}m_{service.date.date()}.html",
    )
    return result


def show_service_calendar_heatmap(workshop: Workshop) -> pd.DataFrame:
    """GTFS有効期間の日別便数を、月×日ヒートマップで示す。"""
    gtfs = workshop.gtfs
    calendar = gtfs.get("calendar")
    exceptions = gtfs.get("calendar_dates")
    starts = []
    ends = []
    if calendar is not None and not calendar.empty:
        starts.append(pd.to_datetime(calendar["start_date"], format="%Y%m%d").min())
        ends.append(pd.to_datetime(calendar["end_date"], format="%Y%m%d").max())
    if exceptions is not None and not exceptions.empty:
        dates = pd.to_datetime(exceptions["date"], format="%Y%m%d")
        starts.append(dates.min())
        ends.append(dates.max())
    if not starts:
        raise ValueError("calendar.txtまたはcalendar_dates.txtがないため運行日を確認できません。")
    start = min(starts)
    end = max(ends)
    counts = gtfs["trips"].groupby("service_id").size()
    rows = []
    for date in pd.date_range(start, end):
        services = _active_service_ids(date, calendar, exceptions)
        rows.append(
            {
                "date": date,
                "trips": int(counts.reindex(list(services)).fillna(0).sum()),
            }
        )
    daily = pd.DataFrame(rows)
    daily["month"] = daily["date"].dt.to_period("M")
    daily["day"] = daily["date"].dt.day
    months = list(pd.period_range(start=start, end=end, freq="M"))
    matrix = np.full((len(months), 31), np.nan)
    month_index = {month: index for index, month in enumerate(months)}
    for row in daily.itertuples():
        matrix[month_index[row.month], row.day - 1] = row.trips

    fig_height = max(4.8, 0.48 * len(months) + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(31), [str(day) for day in range(1, 32)])
    ax.set_yticks(range(len(months)), [str(month) for month in months])
    ax.set_xlabel("日")
    ax.set_ylabel("年月")
    ax.set_title(f"日別運行便数（{start.date()}～{end.date()}）")
    maximum = np.nanmax(matrix) if np.isfinite(matrix).any() else 0
    if len(months) <= 18:
        for row_index in range(len(months)):
            for column_index in range(31):
                value = matrix[row_index, column_index]
                if np.isfinite(value) and value > 0:
                    ax.text(
                        column_index,
                        row_index,
                        str(int(value)),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if value >= maximum * 0.58 else "#222222",
                    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("1日の運行便数")
    plt.tight_layout()
    export = daily.rename(columns={"date": "日付", "trips": "運行便数"})[["日付", "運行便数"]]
    _save_table(export, workshop.output_dir, "service_calendar_daily.csv")
    _save_figure(fig, workshop.output_dir, "service_calendar_heatmap.png")
    plt.show()
    return export


def show_service_calendar(workshop: Workshop) -> pd.DataFrame:
    """旧ノートブック互換の別名。表示内容はヒートマップ。"""
    return show_service_calendar_heatmap(workshop)


def save_results(
    workshop: Workshop,
    service: ServiceResult,
    access: AccessResult,
    map_object,
) -> Path:
    output = workshop.output_dir
    output.mkdir(parents=True, exist_ok=True)
    service.route_summary.to_csv(
        output / f"route_summary_{service.date.date()}.csv", index=False, encoding="utf-8-sig"
    )
    access.coverage_summary.to_csv(
        output / f"coverage_{access.population_year}_{access.distance_m}m.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stops = access.stop_access.copy()
    stops["lon"] = stops.geometry.x
    stops["lat"] = stops.geometry.y
    stops.drop(columns="geometry").to_csv(
        output / f"stop_access_{access.population_year}_{access.distance_m}m.csv",
        index=False,
        encoding="utf-8-sig",
    )
    map_object.save(
        output
        / f"access_map_{access.population_year}_{access.distance_m}m_{service.date.date()}.html"
    )
    settings = pd.DataFrame(
        [
            {"設定": "分析日", "値": str(service.date.date())},
            {"設定": "人口年", "値": access.population_year},
            {"設定": "停留所圏距離（m）", "値": access.distance_m},
        ]
    )
    _save_table(settings, output, "analysis_settings.csv")
    archive_base = output.parent / f"{output.name}_gtfs_gis_results"
    archive = Path(shutil.make_archive(str(archive_base), "zip", output))
    print(f"結果を保存しました：{archive.resolve()}")
    return archive
