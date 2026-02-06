"use client";

import { useCallback, useRef, useState } from "react";
import Map, { MapRef } from "react-map-gl";

const DEFAULT_VIEW_STATE = {
  longitude: -98.5795,
  latitude: 39.8283,
  zoom: 3.5,
};

export default function MapComponent() {
  const mapRef = useRef<MapRef | null>(null);
  const [viewState, setViewState] = useState(DEFAULT_VIEW_STATE);
  const [locating, setLocating] = useState(false);
  const [locateError, setLocateError] = useState<string | null>(null);

  const handleLocate = useCallback(() => {
    setLocateError(null);
    if (!navigator.geolocation) {
      setLocateError("Geolocation is not supported by this browser.");
      return;
    }

    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const zoom = Math.max(mapRef.current?.getZoom() ?? 14, 14);

        setViewState((prev) => ({
          ...prev,
          latitude,
          longitude,
          zoom,
        }));

        mapRef.current?.flyTo({
          center: [longitude, latitude],
          zoom,
          essential: true,
        });

        setLocating(false);
      },
      (error) => {
        setLocateError(error.message || "Unable to retrieve your location.");
        setLocating(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 15000,
      }
    );
  }, []);

  return (
    <div className="fixed inset-0 relative overflow-hidden bg-black">
      <Map
        ref={mapRef}
        mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        viewState={viewState}
        onMove={(event) => setViewState(event.viewState)}
        style={{ width: "100%", height: "100%" }}
      />
      <button
        type="button"
        onClick={handleLocate}
        disabled={locating}
        className="absolute bottom-5 right-5 z-10 flex items-center gap-2 rounded-full bg-white/90 px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg backdrop-blur transition hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-70 dark:bg-slate-900/90 dark:text-white dark:hover:bg-slate-900"
        aria-label="Locate me"
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <circle cx="12" cy="12" r="3" />
          <path d="M2 12h3m14 0h3M12 2v3m0 14v3" />
        </svg>
        <span>{locating ? "Locating..." : "Locate Me"}</span>
      </button>
      {locateError ? (
        <div className="absolute bottom-20 right-5 z-10 max-w-[80vw] rounded-lg bg-black/70 px-3 py-2 text-xs text-white shadow">
          {locateError}
        </div>
      ) : null}
    </div>
  );
}
