from django.shortcuts import render
from movie.models import Movie
from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

def recomendacion(request):
    pelicularecomendada = None

    if request.method =='POST':
    
        prompUsuario = request.POST['searchMovie']
        _ = load_dotenv('openAI.env')
        openai.api_key  = os.environ['openAI_api_key'] 

        emb = get_embedding(prompUsuario,engine='text-embedding-ada-002')
       
        
        peliculas = Movie.objects.all()
        
        resultadoMayor = 0
        for i in peliculas:
            posiblePelicula = i.emb
            print(posiblePelicula)
            resultado = cosine_similarity(emb,posiblePelicula)
            if resultado>resultadoMayor:
                resultadoMayor = resultado
                pelicularecomendada = i
                
        print(pelicularecomendada)

    return render(request, 'recomendacion.html', {'pelicula':pelicularecomendada})
    



