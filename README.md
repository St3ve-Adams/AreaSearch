# AreaSearch - Missing Persons Search Route Generator

A web application that generates optimized driving routes to ensure 100% visual coverage of all accessible roads within a user-defined search area. Built using the **Chinese Postman Problem** (Route Inspection Problem) algorithm.

## The Problem

Unlike standard A-to-B navigation, search and rescue operations require visiting **every street** in an area. This is mathematically known as the **Route Inspection Problem** or **Chinese Postman Problem** - finding the shortest route that traverses every edge (street) in a graph at least once.

## Solution

AreaSearch uses graph theory to:
1. Download the road network from OpenStreetMap
2. Convert it to a mathematical graph (nodes = intersections, edges = streets)
3. "Eulerize" the graph by finding optimal duplicate edges
4. Generate an Eulerian circuit that covers all streets with minimal backtracking

## Features

- Interactive map interface for selecting search areas
- Support for both rectangular and polygon area selection
- Real-time route generation using the Chinese Postman algorithm
- Route statistics (distance, streets covered, repeat segments)
- GPX export for GPS devices
- Dark theme optimized for field use
- Mobile-responsive design

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.10+, FastAPI |
| Algorithm | NetworkX, OSMnx |
| Frontend | HTML5, JavaScript, Leaflet.js |
| Map Data | OpenStreetMap |
| Drawing Tools | Leaflet-Draw |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AreaSearch
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Linux/macOS:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Open your browser** at `http://localhost:8000`

### Alternative: Run directly with uvicorn

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

## Usage

1. **Select Area**: Use the drawing tools (rectangle or polygon) on the map to define your search area
2. **Generate Route**: Click "Plan Route" to generate the optimal search path
3. **View Statistics**: See total distance, number of streets, and repeat segments
4. **Export**: Download the route as a GPX file for your GPS device

## API Reference

### POST `/generate-route`

Generate a search route for the specified area.

**Request Body (Bounding Box)**:
```json
{
  "bbox": {
    "north": 40.7580,
    "south": 40.7480,
    "east": -73.9850,
    "west": -73.9950
  }
}
```

**Request Body (Polygon)**:
```json
{
  "polygon": {
    "coordinates": [
      [-73.99, 40.75],
      [-73.98, 40.75],
      [-73.98, 40.74],
      [-73.99, 40.74],
      [-73.99, 40.75]
    ]
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": "Route generated successfully! Covers 45 street segments.",
  "route": [[lng, lat], ...],
  "total_distance_km": 12.5,
  "num_streets": 45,
  "num_nodes": 32,
  "duplicated_edges": 8,
  "geojson": { ... }
}
```

### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "AreaSearch API"
}
```

## Algorithm Explanation

### The Chinese Postman Problem

The Chinese Postman Problem (CPP) asks: what is the shortest route that visits every edge of a graph at least once and returns to the starting point?

**Key Concepts**:

1. **Eulerian Graph**: A graph where every vertex has an even degree (even number of edges). Such graphs can be traversed visiting each edge exactly once.

2. **Odd-Degree Vertices**: In real road networks, some intersections have an odd number of roads. These must be "fixed" by adding duplicate edges.

3. **Minimum Weight Matching**: We pair up odd-degree vertices optimally to minimize the total extra distance.

### Our Algorithm Steps

1. **Graph Construction**
   - Download road network from OpenStreetMap using OSMnx
   - Convert to undirected graph (for bidirectional streets)
   - Extract the largest connected component

2. **Eulerization**
   - Find all vertices with odd degree
   - Compute shortest paths between all pairs of odd vertices
   - Find minimum weight perfect matching
   - Add duplicate edges along matched paths

3. **Circuit Generation**
   - Apply Hierholzer's algorithm to find Eulerian circuit
   - Convert edge sequence to coordinate list

4. **Optimization**
   - Remove redundant coordinate points
   - Generate GeoJSON for visualization

## Project Structure

```
AreaSearch/
├── backend/
│   └── app.py              # FastAPI application & CPP algorithm
├── frontend/
│   └── index.html          # Web interface (Leaflet + Leaflet-Draw)
├── static/                 # Static assets (if needed)
├── requirements.txt        # Python dependencies
├── run.py                  # Application launcher
├── projectplan.md          # Original project planning document
└── README.md               # This file
```

## Safety Considerations

**Important**: This tool is designed for search and rescue operations.

- **Two-Person Teams**: Always search with a partner - driver watches the road, passenger navigates
- **Legal Compliance**: Follow local laws regarding search operations
- **Coordination**: Coordinate with official search efforts when applicable
- **Communication**: Maintain contact with search coordinators

## Limitations

- **Network Connectivity**: Requires internet to download OSM data (consider caching for field use)
- **Area Size**: Very large areas may take longer to process
- **Road Updates**: Uses OpenStreetMap data which may not reflect recent changes
- **One-Way Streets**: Currently treats all streets as bidirectional

## Future Improvements

- [ ] Offline caching for field operations
- [ ] Multi-vehicle route partitioning
- [ ] Real-time GPS tracking and progress visualization
- [ ] One-way street handling
- [ ] Turn-by-turn directions
- [ ] Route sharing and team coordination

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [OpenStreetMap](https://www.openstreetmap.org/) - Road network data
- [OSMnx](https://github.com/gboeing/osmnx) - Street network analysis
- [NetworkX](https://networkx.org/) - Graph algorithms
- [Leaflet](https://leafletjs.com/) - Interactive maps
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework

---

*Built with care for search and rescue volunteers everywhere.*
