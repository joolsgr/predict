# create a glucose prediction model based on carbohydrate
# and insulin data

class GlucosePredict:
    def __init__(self, carb, insulin):
        self.carb = carb
        self.insulin = insulin

    def predict(self):
        return self.carb * 0.1 + self.insulin * 0.2



