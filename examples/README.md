movielens 20m records: 2000w records 
train 2000w records in 20 seconds. CPU 12 cores

### faiss_als vs. pure als

use recommend() 

```
INFO:implicit:trained model 'als' in 16.075105667114258
INFO:implicit:evaluated model 'als' in 130.9297297000885
INFO:implicit:trained model 'faiss_als' in 16.326430797576904
INFO:implicit:evaluated model 'faiss_als' in 102.83234524726868


,n_intersection@10,n_interaction@10,P@10,R@10,F1@10,n_intersection@5,n_interaction@5,P@5,R@5,F1@5,model
0,1.9308403998,14.8438599878,0.19308404,0.1891927969,0.2078962146,1.1882694507,14.8438599878,0.2376538901,0.1213974727,0.2094081898,als
1,1.66707011,14.8438599878,0.166707011,0.176945121,0.1948293952,1.0663305464,14.8438599878,0.2132661093,0.1163444639,0.2045619961,faiss_als

# recommend loop

INFO:implicit:trained model 'als' in 17.253923654556274
INFO:implicit:evaluated model 'als' in 63.69158601760864
INFO:implicit:trained model 'faiss_als' in 16.350241899490356
INFO:implicit:evaluated model 'faiss_als' in 59.09055972099304
```

faiss(cpu) als has little advantage.
big advantage??? no clue.


use recommend_all()

```
INFO:implicit:trained model 'als' in 16.539251565933228
INFO:implicit:evaluated model 'als' in 40.49670100212097
INFO:implicit:trained model 'faiss_als' in 16.424949884414673
INFO:implicit:evaluated model 'faiss_als' in 40.401150703430176

,n_intersection@10,n_interaction@10,P@10,R@10,F1@10,n_intersection@5,n_interaction@5,P@5,R@5,F1@5,model
0,1.9260956087,14.8438599878,0.1926095609,0.1870386942,0.2075061456,1.1828563791,14.8438599878,0.2365712758,0.1196347119,0.2085802281,als
1,1.9299270832,14.8438599878,0.1929927083,0.1883977047,0.2079770815,1.1886852696,14.8438599878,0.2377370539,0.1211942053,0.2095662815,faiss_als


```

### bm25_weights needed?

turn on

```
,n_intersection@10,n_interaction@10,P@10,R@10,F1@10,n_intersection@5,n_interaction@5,P@5,R@5,F1@5,model
0,0.5502027117,14.8438599878,0.0550202712,0.040736662,0.118630782,0.3279771894,14.8438599878,0.0655954379,0.0248400633,0.1294045267,als
1,0.5477152234,14.8438599878,0.0547715223,0.0408451908,0.1192190858,0.3266109271,14.8438599878,0.0653221854,0.0250191633,0.1304017965,faiss_als

```

turn off

```
,n_intersection@10,n_interaction@10,P@10,R@10,F1@10,n_intersection@5,n_interaction@5,P@5,R@5,F1@5,model
0,1.9263109435,14.8438599878,0.1926310943,0.1882188125,0.2077089553,1.1856111796,14.8438599878,0.2371222359,0.1210382231,0.2093116644,als
1,1.9213136908,14.8438599878,0.1921313691,0.1882195344,0.2073194505,1.1854849488,14.8438599878,0.2370969898,0.1211791523,0.2092018904,faiss_als

```

### faiss

learn how to use faiss in approximate_als.py
