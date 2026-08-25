import json

# Loggings
import logging
import sys
from typing import Any

import httpx2  # is the HTTP client the SDK itself depends on
from mcp.server import MCPServer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


# Initialize MCPServer
mcp = MCPServer("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
GEOCODING_API_BASE = "https://geocoding-api.open-meteo.com/v1/search"
USER_AGENT = "weather-app/1.0"


# Αdd our helper functions for querying and formatting the data from the National Weather Service API
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the National Weather Service API with error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    # asynchronous HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.
    # It can be shared between tasks.
    async with httpx2.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()  # Raise the HTTPStatusError if one occurred.
            return response.json()

        except Exception:
            logger.exception(
                "Failed to fetch NWS data from %s",
                url,
            )
    return None


async def make_geocoding_request(location: str) -> dict[str, Any] | None:
    """Query Open-Meteo Geocoding API for a US location."""

    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
        "countryCode": "US",
    }

    async with httpx2.AsyncClient() as client:
        try:
            response = await client.get(
                GEOCODING_API_BASE,
                params=params,
                timeout=30.0,
            )

            response.raise_for_status()
            return response.json()

        except Exception:
            logger.exception(
                "Failed to geocode location '%s'",
                location,
            )
    return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
        Event: {props.get("event", "Unknown")}
        Area: {props.get("areaDesc", "Unknown")}
        Severity: {props.get("severity", "Unknown")}
        Description: {props.get("description", "No description available")}
        Headline: {props.get("headline", "No headline available")}"
        """
    # Instructions: {props.get("instruction", "No specific instructions provided")}


# Implement tool(services) execution
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """

    url = f"{NWS_API_BASE}/alerts/active?area={state.upper()}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return f"Unable to fetch alerts or no alerts found for state: {state.upper()}"

    alerts = data["features"]
    if not alerts:
        return f"No active alerts for state: {state.upper()}"

    # Avoid excessively large LLM context
    MAX_ALERTS = 10
    selected_alerts = alerts[:MAX_ALERTS]

    formatted_alerts = [format_alert(alert) for alert in selected_alerts]
    return "\n---\n".join(formatted_alerts)


@mcp.tool()
async def get_forecast(lat: float, lon: float) -> str:
    """Get the weather forecast for a specific latitude and longitude.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
    """
    # First get the forecast grid endpoint
    point_url = f"{NWS_API_BASE}/points/{lat},{lon}"
    points_data = await make_nws_request(point_url)

    if not points_data:
        return f"Unable to fetch forecast for coordinates: ({lat}, {lon})"

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    formatted_forecast = []
    for period in periods[:5]:  # Limit to the next 5 periods
        formatted_forecast.append(f"""
            Name: {period["name"]}
            Temperature: {period["temperature"]}°{period["temperatureUnit"]}
            Wind: {period["windSpeed"]} {period["windDirection"]}
            Forecast: {period["detailedForecast"]}
        """)
        # Forecast: {period["shortForecast"]}

    return "\n---\n".join(formatted_forecast)


@mcp.tool()
async def get_location(location: str) -> str:
    """
    Resolve a US city or place name into latitude, longitude,
    state name, and two-letter state code.

    Use this tool when the user provides a city or place name
    instead of latitude/longitude.

    Args:
        location: US city or place name
                  (e.g. "San Francisco", "New York", "Miami").

    Returns:
        JSON containing:
        - location
        - state
        - state_code
        - latitude
        - longitude
    """

    # -----------------------------------------------------
    # 1. Convert location name -> coordinates
    # -----------------------------------------------------

    geocoding_data = await make_geocoding_request(location)

    if not geocoding_data or not geocoding_data.get("results"):
        return json.dumps({"error": f"Unable to find location: {location}"})

    result = geocoding_data["results"][0]

    latitude = result.get("latitude")
    longitude = result.get("longitude")
    resolved_name = result.get("name", location)
    state_name = result.get("admin1")

    if latitude is None or longitude is None:
        return json.dumps({"error": (f"Coordinates unavailable for location: {location}")})

    # Reduce unnecessary coordinate precision
    latitude = round(latitude, 4)
    longitude = round(longitude, 4)

    # -----------------------------------------------------
    # 2. Use NWS metadata to determine state code
    # -----------------------------------------------------

    point_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"

    point_data = await make_nws_request(point_url)

    state_code = None

    if point_data:
        state_code = (
            point_data.get("properties", {})
            .get("relativeLocation", {})
            .get("properties", {})
            .get("state")
        )

    # -----------------------------------------------------
    # 3. Return structured output
    # -----------------------------------------------------

    response = {
        "location": resolved_name,
        "state": state_name,
        "state_code": state_code,
        "latitude": latitude,
        "longitude": longitude,
    }

    # Important:
    # Geocoding may succeed even if NWS state metadata fails.
    if state_code is None:
        response["warning"] = (
            "Location was resolved successfully, but the state code could not be retrieved."
        )

    return json.dumps(
        response,
        indent=2,
    )


# Decorator to register a function as a resource.
@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"


# Prompts are reusable templates that help LLMs interact with your server effectively
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code: \n\n{code}"


# To execute in MCP inspector:
# uv run mcp dev server/weather.py
if __name__ == "__main__":
    mcp.run()  # transport="stdio"
