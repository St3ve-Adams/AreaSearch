const map = L.map("map", { zoomControl: false }).setView(
  [37.7749, -122.4194],
  12
);

L.control.zoom({ position: "bottomright" }).addTo(map);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
}).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const coverageGroup = L.layerGroup().addTo(map);

let areaLayer = null;
let routeLayer = null;

const statusEl = document.getElementById("status");
const spacingInput = document.getElementById("coverage-spacing");
const spacingValue = document.getElementById("spacing-value");
const directionsList = document.getElementById("directions-list");
const totalDistanceEl = document.getElementById("total-distance");
const estimatedTimeEl = document.getElementById("estimated-time");
const searchModeEl = document.getElementById("search-mode");

let activeDrawer = null;

function setStatus(message) {
  statusEl.textContent = message;
}

function clearDirections() {
  directionsList.innerHTML = "<li>Plan a coverage route to see turn-by-turn steps.</li>";
  totalDistanceEl.textContent = "--";
  estimatedTimeEl.textContent = "--";
}

function clearRoute() {
  if (routeLayer) {
    map.removeLayer(routeLayer);
    routeLayer = null;
  }
  coverageGroup.clearLayers();
  clearDirections();
}

function clearAll() {
  clearRoute();
  drawnItems.clearLayers();
  areaLayer = null;
}

function startDrawing() {
  if (activeDrawer) {
    activeDrawer.disable();
  }
  activeDrawer = new L.Draw.Polygon(map, {
    allowIntersection: false,
    showArea: true,
    shapeOptions: {
      color: "#2563eb"
    }
  });
  activeDrawer.enable();
  setStatus("Tap to place points, then double tap to finish.");
}

map.on(L.Draw.Event.CREATED, (event) => {
  drawnItems.clearLayers();
  areaLayer = event.layer;
  drawnItems.addLayer(areaLayer);
  clearRoute();
  map.fitBounds(areaLayer.getBounds(), { padding: [24, 24] });
  setStatus("Area saved. Plan the coverage route.");
});

document.getElementById("draw-area").addEventListener("click", () => {
  startDrawing();
});

document.getElementById("clear-plan").addEventListener("click", () => {
  clearAll();
  setStatus("Cleared. Draw a new search area.");
});

spacingInput.addEventListener("input", (event) => {
  spacingValue.textContent = `${event.target.value} m`;
});

searchModeEl.addEventListener("change", () => {
  if (routeLayer) {
    updateSummary(lastRouteDistance);
  }
});

document.getElementById("plan-route").addEventListener("click", () => {
  if (!areaLayer) {
    setStatus("Draw a search area first.");
    return;
  }

  clearRoute();

  const spacingMeters = Number(spacingInput.value);
  const result = buildCoverageRoute(areaLayer, spacingMeters);

  if (result.routeCoords.length < 2) {
    setStatus("Unable to generate a route. Try increasing spacing.");
    return;
  }

  renderCoverage(result.coverageSegments);
  renderRoute(result.routeCoords);
  const directions = buildDirections(result.routeCoords);
  renderDirections(directions.steps, directions.totalDistance);

  map.fitBounds(routeLayer.getBounds(), { padding: [24, 24] });
  setStatus("Coverage route ready. Start directions when ready.");
});

document.getElementById("start-directions").addEventListener("click", () => {
  if (!routeLayer) {
    setStatus("Plan a coverage route before starting directions.");
    return;
  }
  document.getElementById("directions").scrollIntoView({ behavior: "smooth" });
  setStatus("Directions ready. Follow the steps below.");
});

let lastRouteDistance = 0;

function renderCoverage(segments) {
  coverageGroup.clearLayers();
  segments.forEach((segment) => {
    const latlngs = segment.map((coord) => [coord[1], coord[0]]);
    L.polyline(latlngs, {
      color: "#94a3b8",
      weight: 2,
      dashArray: "4 6",
      opacity: 0.8
    }).addTo(coverageGroup);
  });
}

function renderRoute(routeCoords) {
  const latlngs = routeCoords.map((coord) => [coord[1], coord[0]]);
  routeLayer = L.polyline(latlngs, {
    color: "#2563eb",
    weight: 4
  }).addTo(map);
}

function buildCoverageRoute(polygonLayer, spacingMeters) {
  // Sweep lines across the polygon to approximate full-area coverage.
  const polygonGeo = polygonLayer.toGeoJSON();
  const polygon = turf.polygon(polygonGeo.geometry.coordinates);
  const bbox = turf.bbox(polygon);
  const [minX, minY, maxX, maxY] = bbox;
  const midLat = (minY + maxY) / 2;

  const spacingLat = metersToDegreesLat(spacingMeters);
  const paddingLng = metersToDegreesLng(spacingMeters, midLat);

  const sweepLines = [];

  for (let lat = minY; lat <= maxY + spacingLat / 2; lat += spacingLat) {
    const line = turf.lineString([
      [minX - paddingLng, lat],
      [maxX + paddingLng, lat]
    ]);
    const intersections = turf
      .lineIntersect(line, polygon)
      .features.map((feature) => feature.geometry.coordinates);

    if (intersections.length < 2) {
      continue;
    }

    intersections.sort((a, b) => a[0] - b[0]);
    const segments = [];

    for (let i = 0; i < intersections.length - 1; i += 2) {
      const start = intersections[i];
      const end = intersections[i + 1];
      if (start && end) {
        segments.push([start, end]);
      }
    }

    if (segments.length) {
      sweepLines.push({ lat, segments });
    }
  }

  const routeCoords = [];
  const coverageSegments = [];
  let direction = 1;

  sweepLines.forEach((line) => {
    const ordered = direction === 1
      ? line.segments
      : line.segments.slice().reverse().map((segment) => [segment[1], segment[0]]);

    ordered.forEach((segment) => {
      const [start, end] = segment;
      coverageSegments.push([start, end]);
      const last = routeCoords[routeCoords.length - 1];
      if (!last || last[0] !== start[0] || last[1] !== start[1]) {
        routeCoords.push(start);
      }
      routeCoords.push(end);
    });
    direction *= -1;
  });

  return { routeCoords, coverageSegments };
}

function buildDirections(routeCoords) {
  const steps = [];
  let currentStep = null;
  let totalDistance = 0;

  for (let i = 0; i < routeCoords.length - 1; i += 1) {
    const start = routeCoords[i];
    const end = routeCoords[i + 1];
    const distance = haversineDistance(start, end);

    if (distance < 1) {
      continue;
    }

    totalDistance += distance;
    const bearing = bearingDegrees(start, end);

    if (!currentStep) {
      currentStep = { bearing, distance };
      continue;
    }

    if (angleDifference(bearing, currentStep.bearing) < 25) {
      currentStep.distance += distance;
    } else {
      steps.push(currentStep);
      currentStep = { bearing, distance };
    }
  }

  if (currentStep) {
    steps.push(currentStep);
  }

  return { steps, totalDistance };
}

function renderDirections(steps, totalDistance) {
  directionsList.innerHTML = "";
  if (!steps.length) {
    directionsList.innerHTML = "<li>No directions available. Try a new area.</li>";
  } else {
    steps.forEach((step) => {
      const item = document.createElement("li");
      const direction = bearingToCompass(step.bearing);
      item.textContent = `Head ${direction} for ${formatDistance(step.distance)}.`;
      directionsList.appendChild(item);
    });
  }

  lastRouteDistance = totalDistance;
  updateSummary(totalDistance);
}

function updateSummary(totalDistance) {
  totalDistanceEl.textContent = formatDistance(totalDistance);
  const speed = Number(searchModeEl.value);
  estimatedTimeEl.textContent = formatDuration(totalDistance, speed);
}

function formatDistance(distanceMeters) {
  if (distanceMeters < 1000) {
    return `${Math.round(distanceMeters)} m`;
  }
  return `${(distanceMeters / 1000).toFixed(2)} km`;
}

function formatDuration(distanceMeters, speedKmh) {
  if (!speedKmh || distanceMeters <= 0) {
    return "--";
  }
  const hours = distanceMeters / 1000 / speedKmh;
  const totalMinutes = Math.round(hours * 60);
  const hr = Math.floor(totalMinutes / 60);
  const min = totalMinutes % 60;
  if (hr === 0) {
    return `${min} min`;
  }
  return `${hr} h ${min.toString().padStart(2, "0")} min`;
}

function metersToDegreesLat(meters) {
  return meters / 111320;
}

function metersToDegreesLng(meters, lat) {
  return meters / (111320 * Math.cos((lat * Math.PI) / 180));
}

function haversineDistance(a, b) {
  const R = 6371000;
  const lat1 = (a[1] * Math.PI) / 180;
  const lat2 = (b[1] * Math.PI) / 180;
  const dLat = lat2 - lat1;
  const dLng = ((b[0] - a[0]) * Math.PI) / 180;

  const h =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
  return 2 * R * Math.asin(Math.sqrt(h));
}

function bearingDegrees(a, b) {
  const lat1 = (a[1] * Math.PI) / 180;
  const lat2 = (b[1] * Math.PI) / 180;
  const dLng = ((b[0] - a[0]) * Math.PI) / 180;
  const y = Math.sin(dLng) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
  const bearing = (Math.atan2(y, x) * 180) / Math.PI;
  return (bearing + 360) % 360;
}

function angleDifference(a, b) {
  const diff = Math.abs(a - b) % 360;
  return diff > 180 ? 360 - diff : diff;
}

function bearingToCompass(bearing) {
  const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const index = Math.round(bearing / 45) % 8;
  return directions[index];
}
