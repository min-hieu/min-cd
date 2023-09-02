# min-cd: Correct, Minimal, Fast CD 
original: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/719b0f1ca5ba370616cb837c03ab88d9a88173ff

## Build and Install
```
pip3 install -e git+https://github.com/min-hieu/min-cd.git
```

## min-cd vs. pytorch3d 
```
from min_cd import chamferDist
p1 = torch.rand(1, 10000, 3).to('cuda')
p2 = torch.rand(1, 10000, 3).to('cuda')

cd_func = chamferDist()
mincd_out = cd_func(p1, p2) # output dist1, dist2
py3d_out = pytorch3d.loss.chamfer_distance(p1,p2,'mean') # output dist1.mean() + dist2.mean()

a,b = mincd_out
assert a.mean()+b.mean() == py3d_out
```

## Performance
up to 10x speed compare to py3d
