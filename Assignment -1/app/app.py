from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
from heapq import nlargest

embedding_dicts = {}

for embedding_type in ['glove', 'skipgram', 'skipgram_negative']:
    file_path = f'embed_{embedding_type}.pkl'
    
    with open(file_path, 'rb') as pickle_file:
        embedding_dicts[embedding_type] = pickle.load(pickle_file)

with open('model_gensim.pkl', 'rb') as model_file:
    model_gensim = pickle.load(model_file)

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def similarWords(target_word, embeddings, top_n=10):
    if target_word not in embeddings:
        return ["Word not in Corpus"]

    target_vector = embeddings[target_word]
    cosine_similarities = [
        (word, cosine_similarity(target_vector, embeddings[word]))
        for word in embeddings.keys()
    ]

    top_n_words = nlargest(
        top_n + 1,
        cosine_similarities,
        key=lambda x: x[1]
    )

    top_n_words = [
        word for word, _ in top_n_words if word != target_word
    ]

    return top_n_words[:10]


app = Dash(__name__)

# =========================
# UPDATED LAYOUT (ONLY)
# =========================
app.layout = html.Div(
    style={
        'backgroundColor': '#f4f6f8',
        'minHeight': '100vh',
        'paddingBottom': '40px',
        'fontFamily': "'Segoe UI', Roboto, Arial, sans-serif"
    },
    children=[

        html.H1(
            "Search Engine Page",
            style={
                'textAlign': 'center',
                'fontWeight': '600',
                'letterSpacing': '0.5px',
                'marginTop': '30px',
                'color': '#2c3e50'
            }
        ),

        html.Div(
            style={'marginTop': '40px'},
            children=[

                html.Div(
                    style={
                        'textAlign': 'center',
                        'padding': '30px',
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #e5e7eb',
                        'borderRadius': '12px',
                        'boxShadow': '0 6px 15px rgba(0, 0, 0, 0.08)',
                        'width': '45%',
                        'margin': '0 auto'
                    },
                    children=[

                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': 'GloVe', 'value': 'glove'},
                                {'label': 'Skip-gram', 'value': 'skipgram'},
                                {'label': 'Skip-gram (Negative)', 'value': 'skipgram_negative'},
                            ],
                            placeholder='Select embedding model...',
                            style={
                                'width': '75%',
                                'margin': '0 auto 18px auto',
                                'fontSize': '15px'
                            }
                        ),

                        dcc.Input(
                            id='search-query',
                            type='text',
                            placeholder='Enter a word to explore similarity...',
                            style={
                                'width': '75%',
                                'margin': '0 auto',
                                'padding': '12px',
                                'display': 'block',
                                'borderRadius': '6px',
                                'border': '1px solid #ced4da',
                                'fontSize': '15px'
                            }
                        ),

                        html.Button(
                            'Search',
                            id='search-button',
                            n_clicks=0,
                            style={
                                'padding': '10px 26px',
                                'backgroundColor': '#2563eb',
                                'color': '#ffffff',
                                'border': 'none',
                                'borderRadius': '6px',
                                'marginTop': '22px',
                                'fontSize': '15px',
                                'cursor': 'pointer'
                            }
                        ),
                    ]
                ),
            ]
        ),

        html.Div(
            id='search-results',
            style={
                'marginTop': '40px',
                'padding': '20px',
                'textAlign': 'center',
                'color': '#374151',
                'fontSize': '15px'
            }
        ),
    ]
)

# For displaying the search results
mapping = {
    'skipgram_negative': 'Skipgram (Negative)',
    'glove': 'GloVe',
    'skipgram': 'Skipgram'
}

# Callback to handle search queries
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value'), State('model-selector', 'value')]
)
def search(n_clicks, query, model):
    if n_clicks > 0:
        if not query:
            return html.Div("Please enter a query.", style={'color': 'red'})
        if not model:
            return html.Div("Please select a model from the dropdown.", style={'color': 'red'})
        
        embeddings = embedding_dicts.get(model)
        results = similarWords(query, embeddings)

        return html.Div([
            html.H4(f"Results for '{query}' using model '{mapping[model]}':"),
            html.Ul(
                [html.Li(result) for result in results],
                style={'list-style-type': 'none'}
            )
        ])

    return html.Div(
        "Enter a query and select a model to see results.",
        style={'color': 'gray'}
    )

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
