import requests
url = 'http://10.27.32.183:8111/predict'


r = requests.post(url,json={'user_query':"dht11"})
print(r.json())


print("---------------------------------------------------------")

r = requests.post(url,json={'user_query':"bmp280"})
print(r.json())