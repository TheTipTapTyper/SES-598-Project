import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.clublexus.com%2Fforums%2Fattachments%2Fis-2nd-gen-2006-2013%2F258142d1347371121-pictures-of-you-car-in-the-parking-lot-is250.jpg&f=1&nofb=1'

results = model(img)

results.print()