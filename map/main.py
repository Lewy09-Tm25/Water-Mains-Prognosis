import pickle
from scheduler.core import PlotlyGanttChartTask
import math
import pandas as pd
import plotly.graph_objects as go


def _get_scheduled_tasks() -> tuple[list, list]:
    filehandler = open('../scheduler/CSP/scheduled_tasks.pkl', 'rb')
    csp_scheduled_tasks: list[PlotlyGanttChartTask] = pickle.load(filehandler)
    filehandler.close()

    filehandler = open('../scheduler/LP/scheduled_tasks.pkl', 'rb')
    lp_scheduled_tasks: list[PlotlyGanttChartTask] = pickle.load(filehandler)
    filehandler.close()

    return csp_scheduled_tasks, lp_scheduled_tasks


def _generate_circle_points(center_lat: float, center_lon: float, radius_meters: int, num_points: int) -> list:
    """
    Generates a list of latitude/longitude points forming a circle.
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points  # Angle in radians

        # Haversine formula for calculating latitude and longitude offsets
        lat_offset = math.asin(math.sin(math.radians(center_lat)) * math.cos(radius_meters / 6371000) +
                               math.cos(math.radians(center_lat)) * math.sin(radius_meters / 6371000) * math.cos(angle))
        lon_offset = math.radians(center_lon) + math.atan2(
            math.sin(angle) * math.sin(radius_meters / 6371000) * math.cos(math.radians(center_lat)),
            math.cos(radius_meters / 6371000) - math.sin(math.radians(center_lat)) * math.sin(lat_offset))

        point_lat = math.degrees(lat_offset)
        point_lon = (math.degrees(lon_offset) + 540) % 360 - 180  # Normalize longitude

        points.append((point_lat, point_lon))

    return points


if __name__ == '__main__':
    csp_tasks, lp_tasks = _get_scheduled_tasks()

    center_lat = 39.767826
    center_lon = -104.998209

    prone_pipes = pd.read_csv('../data/risk_prone_pipes.csv')
    ids_of_prone_pipes = list(prone_pipes['Unnamed: 0'])

    worst_pipes = prone_pipes.copy()
    worst_pipes = worst_pipes.sort_values(by='Predicted condition', ascending=True)['Unnamed: 0']
    worst_pipes_ids = list(worst_pipes[:4])

    sa_pipes = pd.read_csv('../data/csp_data.csv')

    geo_points = _generate_circle_points(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_meters=24000,
        num_points=len(ids_of_prone_pipes)
    )

    points = [
        {'lat': lat, 'lon': lon}
        for lat, lon in geo_points
    ]

    scatter_data = []
    for i, id_pipe in zip(range(len(geo_points) - 1), ids_of_prone_pipes):
        if id_pipe in worst_pipes_ids:
            sa_record = sa_pipes[sa_pipes['id_index'] == id_pipe]
            sa_25, sa_55, sa_85 = (
                round(sa_record['25'], 4).values[0],
                round(sa_record['55'], 4).values[0],
                round(sa_record['85'], 4).values[0]
            )
            csp_task = list(filter(lambda task: task['Resource'].split(' ')[-1] == str(id_pipe), csp_tasks))
            lp_task = list(filter(lambda task: task['Resource'].split(' ')[-1] == str(id_pipe), lp_tasks))
            csp_task = pd.DataFrame(csp_task).sort_values(by='Start')
            lp_task = pd.DataFrame(lp_task).sort_values(by='Start')
            csp_emps = ', '.join(list(csp_task['Task'].unique()))
            csp_start = csp_task['Start'].iloc[0].strftime("%m/%d/%Y %H:%M")
            lp_start = lp_task['Start'].iloc[0].strftime("%m/%d/%Y %H:%M")
            hovertemplate=f'Segment ID: {id_pipe}<br><br>Predicted condition: %{{customdata}}<br><br>Survival probability (25 years): {sa_25}<br>Survival probability (55 years): {sa_55}<br>Survival probability (85 years): {sa_85}<br><br>Scheduled for repair (CSP): {csp_start}<br>Workers: {csp_emps}<br><br>Scheduled for repair (LP): {lp_start}<br>Workers: Entire Workforce<extra></extra>'
        else:
            hovertemplate = f'Segment ID: {id_pipe}<br>Predicted condition: %{{customdata}}<extra></extra>'

        scatter_data.append(go.Scattermapbox(
            mode="markers+lines",
            lon=[geo_points[i][1], geo_points[i + 1][1]],
            lat=[geo_points[i][0], geo_points[i + 1][0]],
            marker={'size': 5 if id_pipe not in worst_pipes_ids else 10},
            line={'color': 'grey' if id_pipe not in worst_pipes_ids else 'red'},
            customdata=[round(float(prone_pipes['Predicted condition'][i]), 2)],
            hovertemplate=hovertemplate
        ))
    # Close circle by connecting last point to first
    scatter_data.append(go.Scattermapbox(
        mode="markers+lines",
        lon=[geo_points[-1][1], geo_points[0][1]],
        lat=[geo_points[-1][0], geo_points[0][0]],
        marker={'size': 5},
        line={'color': 'grey'},
        customdata=[round(float(prone_pipes['Predicted condition'][0]), 2)],
        hovertemplate='Info: %{customdata}<extra></extra>'
    ))
    fig = go.Figure(scatter_data)

    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'style': "open-street-map",
            'center': {'lon': center_lon, 'lat': center_lat},
            'zoom': 10},
        showlegend=False
    )

    fig.show()
