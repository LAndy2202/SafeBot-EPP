import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalizar(texto):
    texto = str(texto).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9ñ ]+', ' ', texto).strip()

PALABRAS_DOMINIO = {
    'epp', 'proteccion', 'protector', 'protectora', 'personal', 'industrial', 'seguridad',
    'casco', 'cascos', 'guante', 'guantes', 'bata', 'batas', 'prenda', 'prendas',
    'riesgo', 'riesgos', 'taller', 'accidente', 'accidentes', 'peligro', 'peligros',
    'maquina', 'maquinas', 'mano', 'manos', 'cabeza', 'quimico', 'quimicos',
    'inspeccion', 'inspeccionar', 'vision', 'computacional', 'ia', 'inteligencia',
    'detector', 'deteccion', 'modelo', 'dataset', 'aprendizaje', 'datos', 'imagen',
    'imagenes', 'alerta', 'alertas', 'falso', 'falsa', 'negativo', 'negativa',
    'positivo', 'positiva', 'precision', 'recall', 'map', 'sobreajuste',
    'jerarquia', 'controles', 'equipos', 'equipamiento', 'grieta', 'grietas',
    'rota', 'rotos', 'roto', 'rotas', 'seleccionar', 'elegir', 'usar',
}

class MotorFAQ:
    def __init__(self, ruta='preguntas.csv', umbral=0.23):
        self.datos = pd.read_csv(ruta).fillna('')
        self.umbral = umbral
        self.vectorizador = TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)
        self.matriz = self.vectorizador.fit_transform(self.datos['pregunta'].astype(str))

    def responder(self, pregunta):
        if not normalizar(pregunta):
            return {'respuesta': 'Escribe una pregunta sobre EPP para continuar.', 'fuente': '', 'confianza': 0.0, 'encontrada': False}
        palabras = set(normalizar(pregunta).split())
        if not (palabras & PALABRAS_DOMINIO):
            return {'respuesta': 'Esta consulta está fuera de mi dominio. Solo respondo preguntas sobre equipos de protección personal, prevención de riesgos y detección de EPP mediante IA.', 'fuente': '', 'confianza': 0.0, 'encontrada': False}
        vector = self.vectorizador.transform([pregunta])
        similitudes = cosine_similarity(vector, self.matriz).flatten()
        indice = int(similitudes.argmax())
        puntaje = float(similitudes[indice])
        if puntaje < self.umbral:
            return {'respuesta': 'No tengo una respuesta suficientemente confiable para esa consulta. Puedes preguntarme sobre casco, guantes, bata, selección de EPP o inspección. Para decisiones de seguridad, consulta a un profesional responsable.', 'fuente': '', 'confianza': puntaje, 'encontrada': False}
        fila = self.datos.iloc[indice]
        return {'respuesta': str(fila['respuesta']), 'fuente': str(fila['fuente']), 'confianza': puntaje, 'encontrada': True}
