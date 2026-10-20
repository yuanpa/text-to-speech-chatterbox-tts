import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device = "cpu"
)
texto = "Hola, bienvenidos a Inteligencia artificial generativa, vamos a crear audios, imagenes, texto, y si el pc aguanta videos."

print("Generando audio")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, # Expresion o emocion
    cfg_weight=0.4, # Nos ajusta el ritmo, si la voz va rapida
    temperature=0.8, # Control de la variabilidad, mas altas conlleva una mayor variabilidad, por lo tanto seria menos predecible
    repetition_penalty=1.2, # Penaliza repeticiones extrañas
    min_p=0.5, # Filtra opciones improbables durante la generacion
    top_p=1.0 # Controla el muestre probabilistico, 1.0 lo deja abierto
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado.")
