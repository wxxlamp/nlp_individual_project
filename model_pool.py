from queue import Queue

class ModelPool:
    def __init__(self, model_class, model_name, pool_size=2):
        self.pool = Queue()
        self.model_class = model_class
        self.model_name = model_name
        for _ in range(pool_size):
            self.pool.put(model_class(model_name))

    def get_model(self):
        return self.pool.get()

    def return_model(self, model):
        self.pool.put(model)

    def run(self, premise, hypothesis, method_name):
        model = self.get_model()
        try:
            method = getattr(model, method_name)
            result = method(premise, hypothesis)
        finally:
            self.return_model(model)
        return result
