import json
import time
from main import *
from aiohttp import web
from mrrest import RESTApi
from config import logger


async def sentiment_handler(request):
    try:
        request = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(text='wrong json format')

    text = request.get('text')

    start = time.time()
    result = predict_sentiment(text)
    logger.info('Process time: {:0.2f} sec.'.format(time.time() - start))

    return result


api = RESTApi(
    host='0.0.0.0',
    port=8000,
    routes=[web.post('/predict', sentiment_handler)])

api.run()
