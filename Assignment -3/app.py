import torch
import pickle
import os
import sys
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from torchtext.data.utils import get_tokenizer

# 1. IMPORT ALL COMPONENT FILES
import S2S as S2SFile
import Encoder as EncFile
import Decoder as DecFile
import Encoder_Layer as EncLayerFile
import Mutihead_Attention as AttnFile
import Feed_Forward as FFFile
import Additive_Attention as AdditiveFile

# 2. THE NAMESPACE MAPPING
sys.modules['__main__'].Seq2SeqTransformer = S2SFile.Seq2SeqTransformer
sys.modules['__main__'].Encoder = EncFile.Encoder
sys.modules['__main__'].Decoder = DecFile.Decoder
sys.modules['__main__'].DecoderLayer = DecFile.DecoderLayer
sys.modules['__main__'].EncoderLayer = EncLayerFile.EncoderLayer
sys.modules['__main__'].MultiHeadAttentionLayer = AttnFile.MultiHeadAttentionLayer
sys.modules['__main__'].PositionwiseFeedforwardLayer = FFFile.PositionwiseFeedforwardLayer
sys.modules['__main__'].AdditiveAttention = AdditiveFile.AdditiveAttention

class CallableVocab:
    def __init__(self, v): self.v = v
    def __call__(self, tokens): return [self.v.stoi.get(t, self.v.unk_index) for t in tokens]
    def get_itos(self): return self.v.itos
    @property
    def stoi(self): return self.v.stoi

# 3. GLOBAL CONFIG - FORCE CPU
device = torch.device('cpu') 
SRC_LANG, TARG_LANG = 'en', 'ne'
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, 'model', 'vocab_new.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'general_Seq2SeqTransformer.pt')

with open(VOCAB_PATH, 'rb') as f:
    vocab_transform = pickle.load(f)

# 4. LOAD MODEL & FIX INTERNAL MPS LINKS
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

if isinstance(checkpoint, (list, tuple)):
    params, state = checkpoint[0], checkpoint[1]
    # Rebuild on CPU
    model = S2SFile.Seq2SeqTransformer(**params, device=device).to(device)
    model.load_state_dict(state)
else:
    model = checkpoint.to(device)

# --- CRITICAL FIX FOR MPS ERROR ---
# Manually forcing internal device references to CPU
model.device = device
if hasattr(model, 'encoder'): model.encoder.device = device
if hasattr(model, 'decoder'): model.decoder.device = device
# ----------------------------------

model.eval()

# 5. DASH UI LAYOUT
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H3("A3: English to Nepali Translation App", className='text-center mt-4'), width=6), justify="center"),
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Enter text to translate:"),
            dcc.Input(id='user-input', type='text', className='form-control', placeholder='Type something in English...'),
            html.Br(),
            dbc.Button("Translate", id='translate-btn', color='primary')
        ])
    ], className='mt-4'), width=6), justify="center"),
    dbc.Row(dbc.Col(width=6, id='output-card'), justify="center"),
    
    
], className='mt-5')

# 6. CALLBACK
@app.callback(
    Output('output-card', 'children'),
    Input('translate-btn', 'n_clicks'),
    State('user-input', 'value')
)
def translate_text(n_clicks, text):
    if not n_clicks or n_clicks == 0: return ""
    if not text:
        return dbc.Card([dbc.CardBody([html.P("Please input text.", className='text-muted')])], className='mt-4')

    try:
        # 1. Numericalize
        tokens = en_tokenizer(text.lower())
        src_indices = [vocab_transform[SRC_LANG].stoi['<bos>']] + \
                      vocab_transform[SRC_LANG](tokens) + \
                      [vocab_transform[SRC_LANG].stoi['<eos>']]
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)

        # 2. Greedy Loop
        trg_indices = [vocab_transform[TARG_LANG].stoi['<bos>']]
        for i in range(30):
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                output, _ = model(src_tensor, trg_tensor)
            
            pred_token = output.argmax(2)[:, -1].item()
            trg_indices.append(pred_token)
            if pred_token == vocab_transform[TARG_LANG].stoi['<eos>']:
                break

        # 3. Map to String
        mapping = vocab_transform[TARG_LANG].get_itos()
        res = [mapping[idx] for idx in trg_indices if mapping[idx] not in ['<bos>','<eos>','<pad>','<unk>','[CLS]','[SEP]']]
        final_string = ' '.join(res).replace(" ##", "").replace("##", "")

        return dbc.Card([dbc.CardBody([
            html.H5("Translated Nepali Output:"),
            html.P(children=final_string, style={'fontSize': '24px', 'color': '#2c3e50'})
        ])], className='mt-4')

    except Exception as e:
        return dbc.Card([dbc.CardBody([html.P(f"Error: {str(e)}", className='text-danger')])], className='mt-4')

if __name__ == '__main__':
    app.run(debug=True, port=8050)