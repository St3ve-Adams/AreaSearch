"""
AreaSearch Backend - Missing Persons Search Route Generator

This module implements the Chinese Postman Problem (Route Inspection Problem)
to generate driving routes that traverse every street in a given area.

Author: AreaSearch Team
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from itertools import combinations

import osmnx as ox
import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False

app = FastAPI(
    title="AreaSearch API",
    description="Generate driving routes for missing persons searches using the Chinese Postman Problem algorithm",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BoundingBox(BaseModel):
    """Bounding box coordinates for the search area."""
    north: float = Field(..., description="Northern latitude boundary")
    south: float = Field(..., description="Southern latitude boundary")
    east: float = Field(..., description="Eastern longitude boundary")
    west: float = Field(..., description="Western longitude boundary")


class PolygonArea(BaseModel):
    """Polygon coordinates for custom search area."""
    coordinates: List[List[float]] = Field(..., description="List of [lng, lat] coordinates")


class RouteRequest(BaseModel):
    """Request model for route generation."""
    bbox: Optional[BoundingBox] = None
    polygon: Optional[PolygonArea] = None


class DirectionStep(BaseModel):
    """A single direction step."""
    instruction: str
    street_name: str
    distance_m: float
    coordinates: List[List[float]]  # Segment coordinates
    segment_id: int


class RouteResponse(BaseModel):
    """Response model containing the generated route."""
    success: bool
    message: str
    route: Optional[List[List[float]]] = None  # List of [lng, lat] coordinates
    total_distance_km: Optional[float] = None
    num_streets: Optional[int] = None
    num_nodes: Optional[int] = None
    duplicated_edges: Optional[int] = None
    geojson: Optional[Dict[str, Any]] = None
    directions: Optional[List[DirectionStep]] = None  # Turn-by-turn directions
    segments: Optional[List[Dict[str, Any]]] = None  # Individual street segments for tracking


def download_street_network(bbox: BoundingBox) -> nx.MultiDiGraph:
    """
    Download the driveable street network from OpenStreetMap.

    Args:
        bbox: Bounding box defining the search area

    Returns:
        NetworkX MultiDiGraph representing the street network
    """
    logger.info(f"Downloading street network for bbox: N={bbox.north}, S={bbox.south}, E={bbox.east}, W={bbox.west}")

    # Validate bounding box
    if bbox.north <= bbox.south:
        raise HTTPException(status_code=400, detail="Invalid bounding box: north must be greater than south")
    if bbox.east <= bbox.west:
        raise HTTPException(status_code=400, detail="Invalid bounding box: east must be greater than west")

    # Check if area is too small
    width = bbox.east - bbox.west
    height = bbox.north - bbox.south
    logger.info(f"Bounding box size: {width:.6f} x {height:.6f} degrees")

    if width < 0.0001 or height < 0.0001:
        raise HTTPException(status_code=400, detail="Selected area is too small. Please draw a larger area.")

    # Check if area is too large (max ~5km x 5km = 25 km²)
    # 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 85 km at mid-latitudes
    MAX_DEGREES = 0.05  # ~5 km
    if width > MAX_DEGREES or height > MAX_DEGREES:
        width_km = width * 85  # Approximate
        height_km = height * 111
        raise HTTPException(
            status_code=400,
            detail=f"Selected area is too large ({width_km:.1f} x {height_km:.1f} km). "
                   f"Maximum area is approximately 5 x 5 km. Please select a smaller neighborhood."
        )

    try:
        # Download the drive network using osmnx
        # Note: osmnx 1.x uses bbox parameter as tuple (north, south, east, west)
        G = ox.graph_from_bbox(
            bbox=(bbox.north, bbox.south, bbox.east, bbox.west),
            network_type='drive',
            simplify=True,
            truncate_by_edge=True
        )

        logger.info(f"Downloaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    except Exception as e:
        logger.error(f"Error downloading street network: {e}", exc_info=True)
        error_msg = str(e)
        if "empty" in error_msg.lower() or "no data" in error_msg.lower():
            raise HTTPException(status_code=400, detail="No roads found in the selected area. Try selecting a larger area with streets.")
        raise HTTPException(status_code=400, detail=f"Could not download street network: {error_msg}")


def download_street_network_polygon(coordinates: List[List[float]]) -> nx.MultiDiGraph:
    """
    Download the driveable street network from OpenStreetMap using a polygon.

    Args:
        coordinates: List of [lng, lat] coordinates defining the polygon

    Returns:
        NetworkX MultiDiGraph representing the street network
    """
    logger.info(f"Downloading street network for polygon with {len(coordinates)} vertices")
    logger.info(f"Polygon coordinates: {coordinates[:3]}...")  # Log first few coords

    try:
        from shapely.geometry import Polygon, box
        from shapely.validation import make_valid

        # Validate we have enough points
        if len(coordinates) < 3:
            raise HTTPException(status_code=400, detail="Polygon must have at least 3 vertices")

        # Shapely expects (x, y) = (longitude, latitude) format
        # Frontend sends [lng, lat] which is correct
        polygon_coords = [(coord[0], coord[1]) for coord in coordinates]

        # Ensure polygon is closed
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])

        logger.info(f"Creating polygon with {len(polygon_coords)} points")

        polygon = Polygon(polygon_coords)

        # Validate and fix polygon if needed
        if not polygon.is_valid:
            logger.warning("Polygon is invalid, attempting to fix...")
            polygon = make_valid(polygon)

        if polygon.is_empty:
            raise HTTPException(status_code=400, detail="Polygon is empty or invalid")

        # Check polygon area (rough check - if too small, may have no roads)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        logger.info(f"Polygon bounds: {bounds}, size: {width:.6f} x {height:.6f} degrees")

        if width < 0.0001 or height < 0.0001:
            raise HTTPException(status_code=400, detail="Selected area is too small. Please draw a larger area.")

        # Download the drive network using osmnx
        G = ox.graph_from_polygon(
            polygon,
            network_type='drive',
            simplify=True,
            truncate_by_edge=True
        )

        logger.info(f"Downloaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading street network: {e}", exc_info=True)
        error_msg = str(e)
        if "empty" in error_msg.lower():
            raise HTTPException(status_code=400, detail="No roads found in the selected area. Try selecting a larger area with streets.")
        raise HTTPException(status_code=400, detail=f"Could not download street network: {error_msg}")


def convert_to_undirected_weighted(G: nx.MultiDiGraph) -> nx.MultiGraph:
    """
    Convert directed graph to undirected for Chinese Postman Problem.

    For missing persons searches, we assume all roads are traversable in both directions.
    We keep the minimum edge weight when there are parallel edges.

    Args:
        G: Directed MultiDiGraph from osmnx

    Returns:
        Undirected weighted MultiGraph
    """
    # Create undirected multigraph
    H = nx.MultiGraph()

    # Add all nodes with their attributes
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)

    # Add edges - for parallel edges, keep track of them
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get edge length (distance in meters)
        length = data.get('length', 1)

        # Copy relevant edge attributes
        edge_data = {
            'length': length,
            'weight': length,
            'name': data.get('name', 'Unknown'),
            'highway': data.get('highway', 'unknown'),
            'geometry': data.get('geometry', None)
        }

        H.add_edge(u, v, **edge_data)

    logger.info(f"Converted to undirected graph with {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")
    return H


def get_odd_degree_nodes(G: nx.MultiGraph) -> List[int]:
    """
    Find all nodes with odd degree in the graph.

    For a graph to have an Eulerian circuit (path that visits every edge exactly once
    and returns to the start), all nodes must have even degree.

    Args:
        G: The graph to analyze

    Returns:
        List of node IDs with odd degree
    """
    odd_nodes = [node for node, degree in G.degree() if degree % 2 == 1]
    logger.info(f"Found {len(odd_nodes)} nodes with odd degree")
    return odd_nodes


def get_shortest_paths_between_odd_nodes(G: nx.MultiGraph, odd_nodes: List[int]) -> Dict[Tuple[int, int], Tuple[float, List[int]]]:
    """
    Compute shortest paths between all pairs of odd-degree nodes.

    Args:
        G: The graph
        odd_nodes: List of nodes with odd degree

    Returns:
        Dictionary mapping node pairs to (distance, path) tuples
    """
    logger.info(f"Computing shortest paths between {len(odd_nodes)} odd-degree nodes")

    paths = {}
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i+1:]:
            try:
                path = nx.shortest_path(G, u, v, weight='weight')
                length = nx.shortest_path_length(G, u, v, weight='weight')
                paths[(u, v)] = (length, path)
            except nx.NetworkXNoPath:
                # If no path exists, use infinity
                paths[(u, v)] = (float('inf'), [])

    return paths


def minimum_weight_matching(odd_nodes: List[int], paths: Dict[Tuple[int, int], Tuple[float, List[int]]]) -> List[Tuple[int, int]]:
    """
    Find minimum weight perfect matching for odd-degree nodes.

    This determines which pairs of odd-degree nodes should be connected
    with duplicate edges to make the graph Eulerian.

    Uses a greedy approximation for efficiency.

    Args:
        odd_nodes: List of nodes with odd degree
        paths: Dictionary of shortest paths between odd nodes

    Returns:
        List of node pairs to connect with duplicate edges
    """
    logger.info("Computing minimum weight matching for odd-degree nodes")

    if len(odd_nodes) == 0:
        return []

    if len(odd_nodes) == 2:
        return [(odd_nodes[0], odd_nodes[1])]

    # Create a complete graph of odd nodes with edge weights = shortest path distances
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(odd_nodes)

    for (u, v), (dist, _) in paths.items():
        if dist < float('inf'):
            complete_graph.add_edge(u, v, weight=dist)

    # Use NetworkX's min_weight_matching
    # Note: This finds a minimum weight maximal matching
    try:
        matching = nx.min_weight_matching(complete_graph)
        logger.info(f"Found matching with {len(matching)} pairs")
        return list(matching)
    except Exception as e:
        logger.warning(f"Min weight matching failed: {e}, using greedy approach")
        # Fallback to greedy matching
        return greedy_matching(odd_nodes, paths)


def greedy_matching(odd_nodes: List[int], paths: Dict[Tuple[int, int], Tuple[float, List[int]]]) -> List[Tuple[int, int]]:
    """
    Greedy approximation for minimum weight perfect matching.

    Args:
        odd_nodes: List of nodes with odd degree
        paths: Dictionary of shortest paths between odd nodes

    Returns:
        List of node pairs forming a matching
    """
    remaining = set(odd_nodes)
    matching = []

    # Sort edges by weight
    sorted_edges = sorted(paths.items(), key=lambda x: x[1][0])

    for (u, v), (dist, _) in sorted_edges:
        if u in remaining and v in remaining:
            matching.append((u, v))
            remaining.remove(u)
            remaining.remove(v)

        if len(remaining) == 0:
            break

    return matching


def eulerize_graph(G: nx.MultiGraph) -> Tuple[nx.MultiGraph, int]:
    """
    Make the graph Eulerian by adding duplicate edges.

    This is the core of the Chinese Postman Problem solution.
    We find odd-degree nodes, compute minimum weight matching,
    and add duplicate edges along the shortest paths.

    Args:
        G: The original graph

    Returns:
        Tuple of (Eulerized graph, number of edges added)
    """
    logger.info("Eulerizing the graph...")

    # Work with a copy
    H = G.copy()

    # Find odd-degree nodes
    odd_nodes = get_odd_degree_nodes(H)

    if len(odd_nodes) == 0:
        logger.info("Graph is already Eulerian!")
        return H, 0

    if len(odd_nodes) % 2 != 0:
        logger.warning("Odd number of odd-degree nodes - graph may be disconnected")
        # Try to get the largest connected component
        components = list(nx.connected_components(H))
        if len(components) > 1:
            logger.info(f"Graph has {len(components)} components, using largest")
            largest = max(components, key=len)
            H = H.subgraph(largest).copy()
            odd_nodes = get_odd_degree_nodes(H)

    # Get shortest paths between odd nodes
    paths = get_shortest_paths_between_odd_nodes(H, odd_nodes)

    # Find minimum weight matching
    matching = minimum_weight_matching(odd_nodes, paths)

    # Add duplicate edges along matched paths
    edges_added = 0
    for u, v in matching:
        key = (u, v) if (u, v) in paths else (v, u)
        if key in paths:
            _, path = paths[key]
            # Add edges along the path
            for i in range(len(path) - 1):
                # Get edge data from original graph
                edge_data = H.get_edge_data(path[i], path[i+1])
                if edge_data:
                    # Get the first edge's data (for multigraph)
                    first_key = list(edge_data.keys())[0]
                    data = edge_data[first_key].copy()
                    data['duplicate'] = True
                    H.add_edge(path[i], path[i+1], **data)
                    edges_added += 1

    logger.info(f"Added {edges_added} duplicate edges to Eulerize the graph")

    # Verify the graph is now Eulerian
    remaining_odd = get_odd_degree_nodes(H)
    if len(remaining_odd) > 0:
        logger.warning(f"Graph still has {len(remaining_odd)} odd-degree nodes after Eulerization")

    return H, edges_added


def find_eulerian_circuit(G: nx.MultiGraph) -> List[Tuple[int, int]]:
    """
    Find an Eulerian circuit in the graph.

    Args:
        G: An Eulerian graph (all nodes have even degree)

    Returns:
        List of edges forming the Eulerian circuit
    """
    logger.info("Finding Eulerian circuit...")

    try:
        # Use Hierholzer's algorithm via NetworkX
        circuit = list(nx.eulerian_circuit(G, keys=True))
        logger.info(f"Found Eulerian circuit with {len(circuit)} edges")
        return circuit
    except nx.NetworkXError as e:
        logger.warning(f"Could not find Eulerian circuit: {e}")
        # Fall back to Eulerian path if circuit not possible
        try:
            path = list(nx.eulerian_path(G, keys=True))
            logger.info(f"Found Eulerian path with {len(path)} edges")
            return path
        except nx.NetworkXError:
            logger.error("Could not find Eulerian path either")
            raise HTTPException(status_code=400, detail="Could not generate route - graph may be disconnected")


def circuit_to_coordinates(G: nx.MultiGraph, circuit: List[Tuple[int, int, int]]) -> List[List[float]]:
    """
    Convert Eulerian circuit to a list of coordinates for the route.

    Args:
        G: The graph with node coordinates
        circuit: List of edges (u, v, key) in the circuit

    Returns:
        List of [longitude, latitude] coordinates
    """
    logger.info("Converting circuit to coordinates...")

    coordinates = []

    for u, v, key in circuit:
        # Get edge geometry if available
        edge_data = G.get_edge_data(u, v, key)

        if edge_data and edge_data.get('geometry'):
            # Use the detailed geometry
            geom = edge_data['geometry']
            coords = list(geom.coords)
            # Check if we need to reverse the coordinates
            u_coord = (G.nodes[u]['x'], G.nodes[u]['y'])
            if len(coords) > 0:
                # If the geometry starts closer to v than u, reverse it
                if coords[0] != u_coord:
                    # Check distance to determine if we should reverse
                    dist_to_start = ((coords[0][0] - u_coord[0])**2 + (coords[0][1] - u_coord[1])**2)
                    dist_to_end = ((coords[-1][0] - u_coord[0])**2 + (coords[-1][1] - u_coord[1])**2)
                    if dist_to_end < dist_to_start:
                        coords = coords[::-1]

                for lon, lat in coords:
                    coordinates.append([lon, lat])
        else:
            # Use node coordinates directly
            u_data = G.nodes[u]
            coordinates.append([u_data['x'], u_data['y']])

    # Add the final node
    if circuit:
        last_edge = circuit[-1]
        v_data = G.nodes[last_edge[1]]
        coordinates.append([v_data['x'], v_data['y']])

    # Remove consecutive duplicates
    cleaned_coords = []
    for coord in coordinates:
        if not cleaned_coords or coord != cleaned_coords[-1]:
            cleaned_coords.append(coord)

    logger.info(f"Generated route with {len(cleaned_coords)} coordinate points")
    return cleaned_coords


def coordinates_to_geojson(coordinates: List[List[float]]) -> Dict[str, Any]:
    """
    Convert coordinate list to GeoJSON LineString.

    Args:
        coordinates: List of [longitude, latitude] coordinates

    Returns:
        GeoJSON FeatureCollection
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Search Route",
                    "description": "Generated route for area search"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            }
        ]
    }


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing between two points in degrees."""
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def get_turn_instruction(prev_bearing: float, curr_bearing: float) -> str:
    """Determine turn instruction based on bearing change."""
    diff = (curr_bearing - prev_bearing + 360) % 360
    if diff < 20 or diff > 340:
        return "Continue straight"
    elif 20 <= diff < 70:
        return "Bear right"
    elif 70 <= diff < 110:
        return "Turn right"
    elif 110 <= diff < 160:
        return "Sharp right"
    elif 160 <= diff < 200:
        return "U-turn"
    elif 200 <= diff < 250:
        return "Sharp left"
    elif 250 <= diff < 290:
        return "Turn left"
    else:
        return "Bear left"


def generate_directions(G: nx.MultiGraph, circuit: List[Tuple[int, int, int]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate turn-by-turn directions and segments from the Eulerian circuit.

    Args:
        G: The graph with node coordinates and edge attributes
        circuit: List of edges (u, v, key) in the circuit

    Returns:
        Tuple of (directions list, segments list)
    """
    logger.info("Generating turn-by-turn directions...")

    directions = []
    segments = []
    prev_bearing = None
    prev_street = None
    segment_id = 0

    for i, (u, v, key) in enumerate(circuit):
        edge_data = G.get_edge_data(u, v, key)
        if not edge_data:
            continue

        # Get street name
        street_name = edge_data.get('name', 'Unknown Road')
        if isinstance(street_name, list):
            street_name = street_name[0] if street_name else 'Unknown Road'

        # Get distance
        distance = edge_data.get('length', 0)

        # Get coordinates for this segment
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        if edge_data.get('geometry'):
            geom = edge_data['geometry']
            coords = list(geom.coords)
            u_coord = (u_data['x'], u_data['y'])
            # Check if we need to reverse
            dist_to_start = ((coords[0][0] - u_coord[0])**2 + (coords[0][1] - u_coord[1])**2)
            dist_to_end = ((coords[-1][0] - u_coord[0])**2 + (coords[-1][1] - u_coord[1])**2)
            if dist_to_end < dist_to_start:
                coords = coords[::-1]
            segment_coords = [[c[0], c[1]] for c in coords]
        else:
            segment_coords = [[u_data['x'], u_data['y']], [v_data['x'], v_data['y']]]

        # Calculate bearing for turn instruction
        start_lat, start_lon = segment_coords[0][1], segment_coords[0][0]
        end_lat, end_lon = segment_coords[-1][1], segment_coords[-1][0]
        curr_bearing = calculate_bearing(start_lat, start_lon, end_lat, end_lon)

        # Determine turn instruction
        if prev_bearing is None:
            instruction = f"Start on {street_name}"
        elif street_name != prev_street:
            turn = get_turn_instruction(prev_bearing, curr_bearing)
            instruction = f"{turn} onto {street_name}"
        else:
            instruction = f"Continue on {street_name}"

        # Create segment for tracking
        segment = {
            'id': segment_id,
            'street_name': street_name,
            'distance_m': distance,
            'coordinates': segment_coords,
            'searched': False
        }
        segments.append(segment)

        # Add direction step (consolidate consecutive same-street segments)
        if directions and street_name == prev_street:
            # Extend previous direction
            directions[-1]['distance_m'] += distance
            directions[-1]['coordinates'].extend(segment_coords[1:])  # Skip first point to avoid duplicates
        else:
            direction = {
                'instruction': instruction,
                'street_name': street_name,
                'distance_m': distance,
                'coordinates': segment_coords,
                'segment_id': segment_id
            }
            directions.append(direction)

        prev_bearing = curr_bearing
        prev_street = street_name
        segment_id += 1

    logger.info(f"Generated {len(directions)} direction steps and {len(segments)} segments")
    return directions, segments


def calculate_total_distance(G: nx.MultiGraph, circuit: List[Tuple[int, int, int]]) -> float:
    """
    Calculate total distance of the route in kilometers.

    Args:
        G: The graph
        circuit: The Eulerian circuit

    Returns:
        Total distance in kilometers
    """
    total_meters = 0
    for u, v, key in circuit:
        edge_data = G.get_edge_data(u, v, key)
        if edge_data:
            total_meters += edge_data.get('length', 0)

    return total_meters / 1000  # Convert to km


def generate_chinese_postman_route(bbox: Optional[BoundingBox] = None,
                                    polygon: Optional[PolygonArea] = None) -> RouteResponse:
    """
    Main function to generate the Chinese Postman route for a search area.

    Args:
        bbox: Bounding box defining the search area
        polygon: Polygon defining the search area (alternative to bbox)

    Returns:
        RouteResponse with the generated route
    """
    try:
        # Step 1: Download street network
        if polygon and polygon.coordinates:
            G = download_street_network_polygon(polygon.coordinates)
        elif bbox:
            G = download_street_network(bbox)
        else:
            raise HTTPException(status_code=400, detail="Must provide either bbox or polygon")

        if G.number_of_nodes() == 0:
            return RouteResponse(
                success=False,
                message="No streets found in the specified area",
                route=None
            )

        # Step 2: Convert to undirected graph
        H = convert_to_undirected_weighted(G)

        # Step 3: Get the largest connected component
        if not nx.is_connected(H):
            components = list(nx.connected_components(H))
            logger.info(f"Graph has {len(components)} connected components, using largest")
            largest = max(components, key=len)
            H = H.subgraph(largest).copy()

        original_edges = H.number_of_edges()
        original_nodes = H.number_of_nodes()

        # Step 4: Eulerize the graph
        H_euler, edges_added = eulerize_graph(H)

        # Step 5: Find Eulerian circuit
        circuit = find_eulerian_circuit(H_euler)

        # Step 6: Convert to coordinates
        coordinates = circuit_to_coordinates(H_euler, circuit)

        # Step 7: Calculate statistics
        total_distance = calculate_total_distance(H_euler, circuit)

        # Step 8: Generate directions and segments
        directions, segments = generate_directions(H_euler, circuit)

        # Step 9: Create GeoJSON
        geojson = coordinates_to_geojson(coordinates)

        return RouteResponse(
            success=True,
            message=f"Route generated successfully! Covers {original_edges} street segments.",
            route=coordinates,
            total_distance_km=round(total_distance, 2),
            num_streets=original_edges,
            num_nodes=original_nodes,
            duplicated_edges=edges_added,
            geojson=geojson,
            directions=directions,
            segments=segments
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating route: {str(e)}")


# API Endpoints

@app.get("/")
async def root():
    """Serve the main application page."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "AreaSearch API - Missing Persons Search Route Generator", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AreaSearch API"}


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors."""
    from fastapi.responses import Response
    # Return a simple 1x1 transparent PNG
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/png")


@app.post("/generate-route", response_model=RouteResponse)
async def generate_route(request: RouteRequest):
    """
    Generate a search route for the specified area.

    This endpoint solves the Chinese Postman Problem to create a route
    that covers every driveable street in the area with minimal repetition.

    Args:
        request: RouteRequest containing either a bounding box or polygon

    Returns:
        RouteResponse with the generated route and statistics
    """
    logger.info(f"Received route generation request")

    if not request.bbox and not request.polygon:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'bbox' or 'polygon' in request"
        )

    return generate_chinese_postman_route(bbox=request.bbox, polygon=request.polygon)


@app.post("/generate-route-geojson")
async def generate_route_geojson(request: RouteRequest):
    """
    Generate a search route and return only the GeoJSON.

    This is a convenience endpoint for applications that only need the GeoJSON output.
    """
    response = generate_chinese_postman_route(bbox=request.bbox, polygon=request.polygon)
    if response.success and response.geojson:
        return response.geojson
    raise HTTPException(status_code=400, detail=response.message)


# Mount static files for frontend assets
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Mount frontend directory
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
