-- Returns a list of models that specify the model's attributes
-- Add a model here to train a new model

local models = {}
local model = {}

local parameters = {}
parameters["k"] = 100
parameters["num_of_samples"] = 1000
parameters["hist_size"] = 33
parameters["hist_center"] = 17
parameters["batch_size"] = 256
parameters["epochs"] = 30
parameters["train_ratio"] = 0.5
parameters["learning_rate"] = 0.001



model = {}
model["id"] = "pca"
model["method"] = "pca"
model["name"] = "PCA only"
model["shapes"] = {}
models[model["id"]] = model

model = {}
model["id"] = "re1"
model["method"] = "re"
model["name"] = "Regression (Cube)"
model["shapes"] = {"cube100k"}
model["parameters"] = parameters
models[model["id"]] = model

model = {}
model["id"] = "re2"
model["method"] = "re"
model["name"] = "Regression (Fandisk)"
model["shapes"] = {"fandisk100k"}
model["parameters"] = parameters
models[model["id"]] = model

model = {}
model["id"] = "re3"
model["method"] = "re"
model["name"] = "Regression no noise"
model["shapes"] = {"cube100k", "fandisk100k", "bunny100k", "armadillo100k"}
model["parameters"] = parameters
models[model["id"]] = model

model = {}
model["id"] = "re4"
model["method"] = "re"
model["name"] = "Regression no noise no normalization"
model["shapes"] = {"cube100k", "fandisk100k", "bunny100k", "armadillo100k"}
model["parameters"] = parameters
model["parameters"]["use_num_of_samples"] = true
models[model["id"]] = model


model = {}
model["id"] = "ren1"
model["method"] = "re"
model["name"] = "Regression with noise"
model["comment"] = "brown noise 3e-2"
model["shapes"] = {'cube100k_noise_brown_3e-2','fandisk100k_noise_brown_3e-2','bunny100k_noise_brown_3e-2','armadillo100k_noise_brown_3e-2'}
model["parameters"] = parameters
models[model["id"]] = model

model = {}
model["id"] = "cl1"
model["method"] = "cl"
model["name"] = "Classification no noise"
model["shapes"] = {"cube100k", "fandisk100k", "bunny100k", "armadillo100k"}
model["parameters"] = parameters
model["parameters"]["learning_rate"] = 0.00001
models[model["id"]] = model

model = {}
model["id"] = "cln1"
model["method"] = "cl"
model["name"] = "Classification with noise"
model["comment"] = "brown noise 3e-2"
model["shapes"] = {'cube100k_noise_brown_3e-2','fandisk100k_noise_brown_3e-2','bunny100k_noise_brown_3e-2','armadillo100k_noise_brown_3e-2'}
model["parameters"] = parameters
model["parameters"]["learning_rate"] = 0.00001
models[model["id"]] = model

return models
