from bottle import Bottle, request, response
from network import RNN


class RESTService:
    def __init__(self, host='0.0.0.0', port=33333):
        self._host = host
        self._port = port
        self._model = RNN()
        self._app = Bottle()
        self._route()

    def _route(self):
        self._app.route('/predict',method="POST", callback=self._predict)

    def _predict(self):
        data = request.json
        category = self._model.predict(data)
        response.content_type = 'text/plain'
        return str(category)

    def start(self):
        self._app.run(host=self._host, port=self._port)


server = RESTService()
server.start()