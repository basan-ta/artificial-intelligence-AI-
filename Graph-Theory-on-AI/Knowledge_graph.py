import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import heapq
from collections import deque
from dash.exceptions import PreventUpdate
from base64 import b64decode
import io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Constants
EMPTY, OBSTACLE, START, END, PATH = 0, 1, 2, 3, 4
GRID_SIZE = 20

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_knowledge_graph():
    G = nx.Graph()
    concepts = [
        'AI', 'Machine Learning', 'Deep Learning', 'Computer Vision',
        'NLP', 'Reinforcement Learning', 'Neural Networks', 'CNN', 'RNN',
        'Supervised', 'Unsupervised', 'GANs'
    ]
    edges = [
        ('AI', 'Machine Learning'), ('Machine Learning', 'Deep Learning'),
        ('Deep Learning', 'Neural Networks'), ('Neural Networks', 'CNN'),
        ('Neural Networks', 'RNN'), ('Machine Learning', 'Supervised'),
        ('Machine Learning', 'Unsupervised'), ('Deep Learning', 'GANs'),
        ('AI', 'Computer Vision'), ('AI', 'NLP'),
        ('AI', 'Reinforcement Learning')
    ]
    
    G.add_nodes_from(concepts)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=1, color='#888'),
        hoverinfo='none', mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=20)
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
    
    node_trace.marker.color = [len(list(G.neighbors(node))) for node in G.nodes()]
    
    return go.Figure(data=[edge_trace, node_trace])
def parse_csv(contents):
    content_type, content_string = contents.split(',')
    decoded = b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Store original features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols].fillna(0)
    
    # Standardize and reduce dimensions
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_numeric)
    pca = PCA(n_components=2)
    points = pca.fit_transform(data_scaled)
    
    # Create dataframe with reduced dimensions and original features
    df_pca = pd.DataFrame(points, columns=['x', 'y'])
    df_pca = pd.concat([df_pca, df_numeric], axis=1)
    return df_pca
def create_default_clustering_data(n_clusters=3):
    # Generate synthetic data
    n_samples = 200
    np.random.seed(42)
    centers = np.random.randn(n_clusters, 2) * 3
    X = np.vstack([
        centers[i] + np.random.randn(n_samples // n_clusters, 2) 
        for i in range(n_clusters)
    ])
    return pd.DataFrame(X, columns=['x', 'y'])

def create_clustering_demo(n_clusters=3, data=None):
    if data is None:
        data = create_default_clustering_data(n_clusters)
    
    points = data[['x', 'y']].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(points)
    
    # Create visualization dataframe
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'Cluster': [f'Cluster {l}' for l in cluster_labels]
    })
    
    # Create scatter plot
    fig = px.scatter(df, x='x', y='y',
                    color='Cluster',
                    title='Interactive Clustering Visualization')
    
    # Add centroids
    fig.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(color='black', size=15, symbol='x'),
        name='Centroids'
    ))
    
    # Add cluster characteristics
    for i in range(n_clusters):
        cluster_points = points[cluster_labels == i]
        mean_x, mean_y = cluster_points.mean(axis=0)
        fig.add_annotation(
            x=kmeans.cluster_centers_[i][0],
            y=kmeans.cluster_centers_[i][1],
            text=f'Cluster {i}<br>Size: {len(cluster_points)}',
            showarrow=True,
            arrowhead=1
        )
    
    return fig


def create_path_planning_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    return px.imshow(
        grid,
        color_continuous_scale=['white', 'black', 'green', 'red', 'yellow']
    )

def astar(grid, start, end):
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    rows, cols = grid.shape
    start = tuple(start) if isinstance(start, list) else start
    end = tuple(end) if isinstance(end, list) else end
    
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            next_pos = (current[0] + dx, current[1] + dy)
            
            if (0 <= next_pos[0] < rows and 
                0 <= next_pos[1] < cols and 
                grid[next_pos] != OBSTACLE):
                
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(end, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
    
    return []

app.layout = html.Div([
    html.H1("Graph Theory in AI - Interactive Visualization",
            style={'textAlign': 'center', 'padding': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label='Knowledge Graph', children=[
            html.Div([
                html.H3("AI Knowledge Graph"),
                dcc.Graph(id='knowledge-graph',
                         figure=create_knowledge_graph())
            ])
        ]),
        
        dcc.Tab(label='Clustering', children=[
            html.Div([
                html.H3("Interactive Clustering"),
                html.Div([
                    html.Label('Number of Clusters:'),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2, max=6, value=3, step=1,
                        marks={i: str(i) for i in range(2, 7)}
                    ),
                ], style={'width': '50%', 'margin': '20px auto'}),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or Click to Upload CSV']),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    }
                ),
                dcc.Graph(id='clustering-graph')
            ])
        ]),
        
        dcc.Tab(label='Path Planning', children=[
            html.Div([
                html.H3("A* Path Planning"),
                dbc.ButtonGroup([
                    dbc.Button("Set Start", id="set-start-mode", color="success"),
                    dbc.Button("Set End", id="set-end-mode", color="danger"),
                    dbc.Button("Add Obstacles", id="set-obstacle-mode", color="dark"),
                    dbc.Button("Find Path", id="find-path", color="primary"),
                    dbc.Button("Reset", id="reset-grid", color="secondary"),
                ], style={'margin': '10px'}),
                dcc.Store(id='current-mode', data='obstacle'),
                dcc.Store(id='start-point', data=None),
                dcc.Store(id='end-point', data=None),
                dcc.Graph(id='path-planning-grid',
                         figure=create_path_planning_grid())
            ])
        ])
    ])
])

@app.callback(
    Output('clustering-graph', 'figure'),
    [Input('cluster-slider', 'value'),
     Input('upload-data', 'contents')]
)
def update_clustering(n_clusters, contents):
    if contents is None:
        return create_clustering_demo(n_clusters)
    try:
        df = parse_csv(contents)
        return create_clustering_demo(n_clusters, df)
    except Exception as e:
        return create_clustering_demo(n_clusters)

@app.callback(
    Output('current-mode', 'data'),
    [Input('set-start-mode', 'n_clicks'),
     Input('set-end-mode', 'n_clicks'),
     Input('set-obstacle-mode', 'n_clicks')]
)
def update_mode(start_clicks, end_clicks, obstacle_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'obstacle'
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    mode_map = {
        'set-start-mode': 'start',
        'set-end-mode': 'end',
        'set-obstacle-mode': 'obstacle'
    }
    return mode_map[button_id]

@app.callback(
    [Output('path-planning-grid', 'figure'),
     Output('start-point', 'data'),
     Output('end-point', 'data')],
    [Input('path-planning-grid', 'clickData'),
     Input('reset-grid', 'n_clicks'),
     Input('find-path', 'n_clicks')],
    [State('current-mode', 'data'),
     State('path-planning-grid', 'figure'),
     State('start-point', 'data'),
     State('end-point', 'data')]
)
def update_grid(click_data, reset_clicks, find_clicks, 
                mode, figure, start_point, end_point):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    grid = np.array(figure['data'][0]['z'])

    if button_id == 'reset-grid':
        return create_path_planning_grid(), None, None

    if button_id == 'path-planning-grid' and click_data:
        y, x = click_data['points'][0]['y'], click_data['points'][0]['x']
        pos = tuple((int(y), int(x)))  # Convert to tuple
        
        if mode == 'start':
            if start_point:
                grid[start_point[0]][start_point[1]] = EMPTY
            grid[pos[0]][pos[1]] = START
            start_point = pos
        elif mode == 'end':
            if end_point:
                grid[end_point[0]][end_point[1]] = EMPTY
            grid[pos[0]][pos[1]] = END
            end_point = pos
        elif mode == 'obstacle':
            current_val = grid[pos[0]][pos[1]]
            if current_val not in [START, END]:  # Don't override start/end points
                grid[pos[0]][pos[1]] = EMPTY if current_val == OBSTACLE else OBSTACLE

    if button_id == 'find-path' and start_point and end_point:
        # Clear previous path
        grid[(grid == PATH)] = EMPTY
        # Find new path
        path = astar(grid, start_point, end_point)
        if path:
            # Mark path excluding start and end points
            for pos in path[1:-1]:
                grid[pos[0]][pos[1]] = PATH

    # Update figure with new grid
    figure['data'][0]['z'] = grid
    return figure, start_point, end_point


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)