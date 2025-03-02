import configparser
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
import geopy.distance
import geopandas as gpd
import osmnx as ox
from dataclasses import dataclass, field


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized.")

def load_config() -> configparser.ConfigParser:
    """
    Loads the API configuration from the config file.
    
    Returns:
        configparser.ConfigParser: Content of the config file. 
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

@dataclass
class TramData:
    """
    A dataclass containing all the stops and lines data. 
    """
    stops_df: pd.DataFrame = None
    lines_df: pd.DataFrame = None

    def __post_init__(self):
        logger.info("Initializing TramData...")
        self.mpk_sourcing = MpkSourcing()
        self.mpk_sourcing.fetch_and_process()

        self.stops_df = self._create_stops_df()
        self.lines_df = self._create_lines_df()

    @cached_property
    def stops_data(self) -> Dict:
        """
        Fetches and caches stops data from MpkSourcing.

        Returns:
            Dict: with stop names as keys and their details as values.
        """
        logger.info("Fetching stops data...")
        stop_data = {}
        for stop in self.mpk_sourcing.stops_data:
            stop_name = stop["name"]
            passages_info = self.mpk_sourcing.passages_data.get(stop_name, {})
            stop_data[stop_name] = {
                "category": stop["category"],
                "latitude": stop["lat"],
                "longitude": stop["lng"],
                "number_of_lines": passages_info.get("number_of_lines", None),
                "lines": passages_info.get("lines", []),
                "min_time_between_trams": passages_info.get("min_time_between_trams", None),
                "max_time_between_trams": passages_info.get("max_time_between_trams", None),
                "avg_time_between_trams": passages_info.get("avg_time_between_trams", None)
            }
        return stop_data

    def _create_stops_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame with stop data and passages data.
        
        Returns:
            pd.DataFrame: DataFrame containing stop details and passage data.
        """
        logger.info("Creating stops DataFrame...")
        stop_data = []
        for stop_name, data in self.stops_data.items():
            stop_data.append({
                "stop": stop_name,
                "category": data["category"],
                "latitude": data["latitude"],
                "longitude": data["longitude"],
                "number_of_lines": data["number_of_lines"],
                "lines": data["lines"],
                "min_time_between_trams": data["min_time_between_trams"],
                "max_time_between_trams": data["max_time_between_trams"],
                "avg_time_between_trams": data["avg_time_between_trams"]
            })
        stops_df = pd.DataFrame(stop_data)
        stops_df.set_index("stop", inplace=True)
        logger.info("Stops DataFrame created successfully.")
        return stops_df

    def _create_lines_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame with line-to-stop data and coordinates of stops.
        
        Returns:
            pd.DataFrame: DataFrame containing line details and coordinates.
        """
        logger.info("Creating lines DataFrame...")
        lines_data = []
        for line, stops in self.mpk_sourcing.lines_to_stops.items():
            coordinates = self.mpk_sourcing.lines_stops_coordinates.get(line, [])
            lines_data.append({
                "line": line,
                "stops": stops,
                "coordinates": coordinates
            })
        lines_df = pd.DataFrame(lines_data)
        lines_df.set_index("line", inplace=True)
        logger.info("Lines DataFrame created successfully.")
        return lines_df
    

class MpkSourcing:
    def __init__(self):
        self.stops_data: List[Dict] = []
        self.passages_data: Dict[str, Dict] = {}
        self.lines_to_stops: Dict[int, List[str]] = {}
        self.lines_stops_coordinates: Dict[int, List[tuple]] = {}
        self.config = load_config()

    def get_all_stops(self) -> List[Dict]:
        """
        Fetches all stops from the MPK API.

        Returns:
            List[Dict]: List of dictionaries containing stop data.
        """
        logger.info("Fetching stops data from API...")
        response = requests.get(self.config['API']['STOPS_URL'], params=self.get_params())
        if response.status_code == 200:
            logger.info("Successfully fetched stops data.")
            return response.json().get("stops", [])
        else:
            logger.error(f"Error fetching stops data. Response code: {response.status_code}")
            return []

    def get_stop_passages(self, stop_id: str) -> Dict:
        """
        Fetches passage data for a specific stop.

        Args:
            stop_id (str): The stop's ID.

        Returns:
            Dict: A dictionary containing line and passage information for the stop.
        """
        logger.info(f"Fetching passage data for stop: {stop_id}")
        params = {"stop": stop_id}
        response = requests.get(self.config['API']['PASSAGES_URL'], params=params)
        if response.status_code == 200:
            data = response.json()
            relative_times = [
                entry["actualRelativeTime"] for entry in data.get("actual", [])
            ] + [
                entry["actualRelativeTime"] for entry in data.get("old", [])
            ]
            return {
                "number_of_lines": len(data.get('routes', [])),
                "lines": [int(route['name']) for route in data.get('routes', [])],
                "min_time_between_trams": min(relative_times, default=None),
                "max_time_between_trams": max(relative_times, default=None),
                "avg_time_between_trams": np.mean(relative_times) if relative_times else None
            }
        else:
            logger.error(f"Error fetching passages for stop {stop_id}")
            return {}

    def get_params(self) -> Dict:
        """Returns the parameters for the API request."""
        return {
            "left": int(self.config['PARAMS']['left']),
            "bottom": int(self.config['PARAMS']['bottom']),
            "right": int(self.config['PARAMS']['right']),
            "top": int(self.config['PARAMS']['top'])
        }

    def fetch_stops_and_passages(self):
        """Fetches all stops and their passage data concurrently."""
        logger.info("Starting fetch for stops and passages...")
        convert_to_angled_lat_lng = lambda x: float(x) * 2.7778e-07

        stops = self.get_all_stops()
        self.stops_data = [
            {
                "name": stop["name"], 
                "lat": convert_to_angled_lat_lng(stop["latitude"]), 
                "lng": convert_to_angled_lat_lng(stop["longitude"]), 
                'category': stop['category']
            }
            for stop in stops
        ]
        
        with ThreadPoolExecutor() as executor:
            future_to_stop = {executor.submit(self.get_stop_passages, stop["shortName"]): stop for stop in stops}
            for future in future_to_stop:
                stop = future_to_stop[future]
                try:
                    result = future.result()
                    self.passages_data[stop["name"]] = result
                except Exception as exc:
                    logger.error(f"Error fetching passages for stop {stop['name']}: {exc}")
                time.sleep(0.5)  

    def process_lines_to_stops(self):
        """Processes the mapping of lines to their corresponding stops."""
        logger.info("Processing line-to-stop mappings...")
        for stop, data in self.passages_data.items():
            lines = data['lines']
            for line in lines:
                if line not in self.lines_to_stops:
                    self.lines_to_stops[line] = []
                self.lines_to_stops[line].append(stop)

    @staticmethod
    def find_closest_point(start_point: tuple, points: List[tuple]) -> tuple:
        """
        Finds the closest point from the list of points to the start point.

        Args:
            start_point (tuple): The starting point coordinates (latitude, longitude).
            points (List[tuple]): List of coordinates (latitude, longitude) to search through.

        Returns:
            tuple: The closest point from the list of points.
        """
        closest_point = None
        min_distance = float('inf')
        for point in points:
            dist = geopy.distance.distance(start_point, point).km
            if dist < min_distance:
                min_distance = dist
                closest_point = point
        return closest_point

    def process_lines_stops_coordinates(self):
        """Processes the ordered coordinates of each line's stops."""
        logger.info("Processing line stop coordinates...")

        for line, stops_lines in self.lines_to_stops.items():
            coordinates = [
                (stop['lat'], stop['lng'])
                for stop in self.stops_data if stop['name'] in stops_lines
            ]

            ordered_coordinates = [coordinates[0]]
            remaining_points = coordinates[1:]

            while remaining_points:
                last_point = ordered_coordinates[-1]
                closest_point = self.find_closest_point(last_point, remaining_points)
                ordered_coordinates.append(closest_point)
                remaining_points.remove(closest_point)

            self.lines_stops_coordinates[line] = ordered_coordinates

    def fetch_and_process(self):
        """Fetches and processes all necessary data (stops, passages, lines, coordinates)."""
        self.fetch_stops_and_passages()
        self.process_lines_to_stops()
        self.process_lines_stops_coordinates()
        logger.info("Data fetching and processing completed.")

@dataclass
class OpenStreetMapData:
    """Class for retrieving OpenStreetMap (OSM) data, including buildings and streets.

    Attributes:
        place (str): Name of the location to retrieve data for.
        buildings_df (gpd.GeoDataFrame): GeoDataFrame containing building geometries.
        streets_df (gpd.GeoDataFrame): GeoDataFrame containing street geometries.
    """

    place: str = "Kraków, Poland"
    buildings_df: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    streets_df: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)

    def __post_init__(self) -> None:
        """Initializes the OpenStreetMapData class by retrieving buildings and streets."""
        logging.info(f"Initializing OSM data retrieval for: {self.place}")
        try:
            self._fetch_buildings()
            self._fetch_streets()
        except Exception as e:
            logging.error(f"Error initializing OSM data: {e}")

    def _fetch_buildings(self) -> None:
        """Fetches building geometries from OSM and computes their centroids."""
        self.buildings_df = ox.geometries_from_place(
            self.place, tags={"building": True}
        )[['geometry']].dropna()
        logging.info(f"Successfully retrieved {len(self.buildings_df)} buildings.")

        self.buildings_df = self._calculate_centroid(self.buildings_df)

    def _fetch_streets(self) -> None:
        """Fetches street geometries from OSM as a GeoDataFrame."""
        graph = ox.graph_from_place(self.place, network_type="all")
        self.streets_df = ox.graph_to_gdfs(graph, nodes=False)
        logging.info(f"Successfully retrieved {len(self.streets_df)} street segments.")

    @staticmethod
    def _calculate_centroid(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculates the centroid of each building polygon.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing building geometries.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with centroid coordinates.
        """
        df = df.copy()
        df["centroid"] = df.geometry.centroid
        df["lat"] = df["centroid"].y
        df["lng"] = df["centroid"].x
        return df.drop(columns=["centroid"])
