"""Motor de preguntas frecuentes SafeBot-EPP (recuperación TF-IDF).
No genera respuestas nuevas: selecciona contenido del CSV.
"""
import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalizar(texto):
    texto = str(texto).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9ñ ]+', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

PALABRAS_DOMINIO = {
    'epp', 'proteccion', 'protector', 'protectora', 'personal', 'industrial', 'seguridad',
    'casco', 'cascos', 'guante', 'guantes', 'bata', 'batas', 'prenda', 'prendas',
    'riesgo', 'riesgos', 'taller', 'accidente', 'accidentes', 'peligro', 'peligros',
    'maquina', 'maquinas', 'mano', 'manos', 'cabeza', 'quimico', 'quimicos',
    'inspeccion', 'inspeccionar', 'vision', 'computacional', 'ia', 'inteligencia',
    'detector', 'deteccion', 'modelo', 'dataset', 'aprendizaje', 'datos', 'imagen',
    'imagenes', 'alerta', 'alertas', 'falso', 'falsa', 'negativo', 'negativa',
    'positivo', 'positiva', 'precision', 'recall', 'map', 'sobreajuste', 'jerarquia',
    'controles', 'equipos', 'equipamiento', 'grieta', 'grietas', 'rota', 'rotos',
    'roto', 'rotas', 'seleccionar', 'elegir', 'usar', 'iou', 'f1', 'fps', 'inferencia',
    'fotograma', 'fotogramas', 'clasificacion', 'guante', 'guantes'
}

MENSAJE_FUERA = ('Esta consulta está fuera de mi dominio. Solo respondo preguntas sobre '
                 'equipos de protección personal, prevención de riesgos y detección de EPP mediante IA.')
MENSAJE_DUDA = ('No tengo una respuesta suficientemente confiable para esa consulta. '
               'Puedes preguntarme sobre casco, guantes, bata, selección de EPP o inspección. '
               'Para decisiones de seguridad, consulta a un profesional responsable.')

class MotorFAQ:
    def __init__(self, ruta='preguntas.csv', umbral=0.23):
        self.datos = pd.read_csv(ruta).fillna('')
        self.umbral = umbral
        self.preguntas_norm = self.datos['pregunta'].astype(str).map(normalizar).tolist()
        self.exactas = {q: i for i, q in enumerate(self.preguntas_norm)}
        self.vectorizador = TfidfVectorizer(preprocessor=normalizar,
                                            ngram_range=(1, 2), sublinear_tf=True)
        self.matriz = self.vectorizador.fit_transform(self.datos['pregunta'].astype(str))

    def _resultado(self, indice, puntaje):
        fila = self.datos.iloc[indice]
        return {'respuesta': str(fila['respuesta']), 'fuente': str(fila['fuente']),
                'confianza': puntaje, 'encontrada': True}

    def _resultado_especial(self, consulta):
        # Guardas de concepto: evita mezclar falsos positivos y falsos negativos.
        ambos = ('falso positivo' in consulta and 'falso negativo' in consulta)
        ambos_metricas = ('precision' in consulta and 'recall' in consulta)
        if ambos:
            return 'cual es la diferencia entre un falso positivo y un falso negativo'
        if ambos_metricas:
            return 'cual es la diferencia entre precision y recall'
        if ('falso positivo' in consulta or 'falsa alarma' in consulta) and 'falso negativo' not in consulta:
            return 'que es un falso positivo'
        if 'falso negativo' in consulta and 'falso positivo' not in consulta:
            return 'que es un falso negativo'
        # Prioriza la intención de detectar cascos, no la de verificar batas.
        if ('casco' in consulta or 'cascos' in consulta) and (
            'ia' in consulta.split() or 'inteligencia artificial' in consulta or
            'vision computacional' in consulta or 'vision artificial' in consulta
        ) and any(x in consulta for x in ('detect', 'reconoc', 'identific')):
            if any(x in consulta for x in ('bien coloc', 'correctamente', 'bien puesto', 'uso correcto')):
                return 'si la ia detecta un casco significa que esta bien colocado'
            return 'la inteligencia artificial puede detectar un casco'
        return None

    def responder(self, pregunta):
        consulta = normalizar(pregunta)
        if not consulta:
            return {'respuesta': 'Escribe una pregunta sobre EPP para continuar.',
                    'fuente': '', 'confianza': 0.0, 'encontrada': False}
        palabras = set(consulta.split())
        if not (palabras & PALABRAS_DOMINIO):
            return {'respuesta': MENSAJE_FUERA, 'fuente': '',
                    'confianza': 0.0, 'encontrada': False}
        # Exact match is the strongest evidence for FAQ retrieval.
        if consulta in self.exactas:
            return self._resultado(self.exactas[consulta], 1.0)
        clave = self._resultado_especial(consulta)
        if clave and clave in self.exactas:
            return self._resultado(self.exactas[clave], 1.0)
        vector = self.vectorizador.transform([pregunta])
        similitudes = cosine_similarity(vector, self.matriz).flatten()
        indice = int(similitudes.argmax())
        puntaje = float(similitudes[indice])
        if puntaje < self.umbral:
            return {'respuesta': MENSAJE_DUDA, 'fuente': '',
                    'confianza': puntaje, 'encontrada': False}
        return self._resultado(indice, puntaje)
