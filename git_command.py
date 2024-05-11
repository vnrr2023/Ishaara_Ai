import os

mssg=input("enter mssg:")
os.system('git add .')
os.system('git commit -m'+'"'+f'{mssg}'+'"')
os.system('git push')