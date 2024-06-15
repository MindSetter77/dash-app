# -*- coding: utf-8 -*-
import os
import io
import cv2
import base64
import numpy as np
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from tensorflow.keras import models
import joblib
from PIL import Image

# Inicjalizacja aplikacji Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Przeglądarka obrazów"

CNN = True

# Klasa przeglądarki obrazów
class ImageBrowser:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = os.listdir(folder_path)
        self.current_index = 0

    def get_current_image_name(self):
        return self.images[self.current_index]

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)

    def previous_image(self):
        self.current_index = (self.current_index - 1) % len(self.images)

# Wczytanie modelu
model = models.load_model('E12v64.keras')
knn_model = joblib.load('knn_model.joblib')


# Layout aplikacji
app.layout = dbc.Container([
    html.H1("Przeglądarka obrazów", className="text-center"),
    dcc.Store(id='image-index-store', data={'index': 0}),
    dcc.Dropdown(
        id='dropdown-menu',
        options=[
            {'label': 'CNN', 'value': 'CNN'},
            {'label': 'KNN', 'value': 'KNN'}
        ],
        value='CNN',
        style={'margin-bottom': '20px'}  # Dodanie marginesu dolnego
    ),
    html.Div(id='wybrana-opcja'),
    dbc.Row([
        dbc.Col(html.Button("Previous", id='previous-image-button', n_clicks=0), width=2),
        dbc.Col(html.Div(id='image-container', style={'height': '400px', 'overflow': 'hidden', 'display': 'flex', 'align-items': 'center'}), width=8),
        dbc.Col(html.Button("Next", id='next-image-button', n_clicks=0), width=2),
    ], style={'background-color': '#f0f0f0'}),  
    dbc.Row([
        dbc.Col(html.Button("Rozpoznaj zwierze", id='predict-button', n_clicks=0, className="mt-3"), width=12, style={'text-align': 'center'})
    ], className="mt-2"),  # Nowy wiersz bez marginesu
    dbc.Row([
        html.Div(id='animal-name-container', className="mt-3 text-center"),  # Wyśrodkowanie napisu
        html.Div(id='prediction-label-container', className="text-center")   # Wyśrodkowanie napisu
    ], justify='center'),  # Wyśrodkowanie wiersza

    # Dodane pole do przeciągania zdjęcia
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Przeciągnij pliki tutaj lub ',
            html.A('wybierz pliki')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        # Dozwolone rozszerzenia plików
        # 'accept': '.jpg, .jpeg, .png'
    ),
    dbc.Row([
        html.Div("asd", id='container', className="mt-3 text-center"),  # Wyśrodkowanie napisu
    ])
], className="mt-5")


# Callback do wczytywania obrazu
@app.callback(
    Output('image-container', 'children'),
    [Input('previous-image-button', 'n_clicks'),
     Input('next-image-button', 'n_clicks')]
)
def update_image(previous_clicks, next_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'previous-image-button':
        browser.previous_image()
    elif button_id == 'next-image-button':
        browser.next_image()

    image_name = browser.get_current_image_name()
    image_path = os.path.join("test", image_name)
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    return html.Img(src=f"data:image/jpeg;base64,{encoded_string}", style={'width': '100%', 'max-height': '500px'})

# Callback do przewidywania zwierzęcia
@app.callback(
    [Output('animal-name-container', 'children'),
     Output('prediction-label-container', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('image-container', 'children')]
)
def predict_animal(n_clicks, image):
    if n_clicks:
        image_name = browser.get_current_image_name()
        image_path = os.path.join("test", image_name)

        if(CNN):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))

            prediction = model.predict(np.array([img]) / 255)
            class_names = ['Kon', 'Krowa', 'Kura', 'Owca', 'Pies', 'Motyl', 'Kot', 'Pajak', 'Slon', 'Wiewiorka']
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index] * 100

            return f"Na obrazku jest: {class_names[predicted_class_index]}", f"Pewność: {confidence:.2f}%"
        else:
            # KNN Prediction
            image = Image.open(image_path)
            desired_shape = (64, 64)
            image = image.resize(desired_shape)
            photo_array = np.array(image)
            if photo_array.shape == (64, 64, 3):
                flattened_image = photo_array.flatten()

                # Predykcja na pojedynczym obrazie
                predicted_label = knn_model.predict([flattened_image])
                class_names = ['Kon', 'Krowa', 'Kura', 'Owca', 'Pies', 'Motyl', 'Kot', 'Pajak', 'Slon', 'Wiewiorka']

                # Wyświetlenie przewidzianej etykiety
                print("Przewidziana etykieta:", class_names[predicted_label[0]])

                return f"Na obrazku jest: {class_names[predicted_label[0]]}", "Model KNN"
    return "", ""

@app.callback(
    Output('container', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def imageSend(contents, filename):
    if contents is not None:
        # Dekodowanie przesłanego obrazu
        decoded_image = base64.b64decode(contents.split(',')[1])
        img = Image.open(io.BytesIO(decoded_image))

        # Przetworzenie obrazu
        if CNN:
            img_array = np.array(img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Konwersja do formatu BGR (OpenCV)
            img_array = cv2.resize(img_array, (64, 64))

            # Predykcja na obrazie
            prediction = model.predict(np.array([img_array]) / 255)
            class_names = ['Kon', 'Krowa', 'Kura', 'Owca', 'Pies', 'Motyl', 'Kot', 'Pajak', 'Slon', 'Wiewiorka']
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index] * 100

            # Zwrócenie wyników predykcji jako zawartość kontenera
            return [
                html.Div(f"Na obrazku jest: {class_names[predicted_class_index]}"),
                html.Div(f"Pewność: {confidence:.2f}%")
            ]
        else:
            # Przetwarzanie i predykcja na obrazie za pomocą modelu KNN
            img_array = np.array(img.resize((64, 64)))
            if img_array.shape == (64, 64, 3):
                flattened_image = img_array.flatten()

                # Predykcja na pojedynczym obrazie
                predicted_label = knn_model.predict([flattened_image])
                class_names = ['Kon', 'Krowa', 'Kura', 'Owca', 'Pies', 'Motyl', 'Kot', 'Pajak', 'Slon', 'Wiewiorka']

                # Zwrócenie wyników predykcji jako zawartość kontenera
                return [
                    html.Div(f"Na obrazku jest: {class_names[predicted_label[0]]}"),
                    html.Div("Model KNN")
                ]
    else:
        # Jeśli nie ma przesłanej zawartości, zwróć pusty kontener
        return html.Div()

    
@app.callback(
    [],
    [Input('dropdown-menu', 'value')]
)
def update_dropdown_value(selected_option):
    global CNN
    if selected_option == 'CNN':
        CNN = True
    elif selected_option == 'KNN':
        CNN = False

    print("Stan zmiennej CNN:", CNN)  # Wydrukuj stan zmiennej CNN



# Utworzenie przeglądarki obrazów
browser = ImageBrowser("test")

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
