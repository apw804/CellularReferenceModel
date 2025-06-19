# Keith Briggs 2023-07-17
# Generate points of a hexagonal lattice in a spiral order
# Quick check:
#   python3 hexagon_lattice_generator_02.py | graph -TX -m0 -S 16
# Make nice pdf picture:
#   python3 hexagon_lattice_generator_02.py pdf
# Check no duplicate points:
#   python3 hexagon_lattice_generator_02.py | wc -l
#   python3 hexagon_lattice_generator_02.py | sort | uniq | wc -l

# ''' https://oeis.org/A028896
#                  85--84--83--82--81--80
#                  /                     \
#                86  56--55--54--53--52  79
#                /   /                 \   \
#              87  57  33--32--31--30  51  78
#              /   /   /             \   \   \
#            88  58  34  16--15--14  29  50  77
#            /   /   /   /         \   \   \   \
#          89  59  35  17   5---4  13  28  49  76
#          /   /   /   /   /     \   \   \   \   \
#     <==90==60==36==18===6===0   3  12  27  48  75
#            /   /   /   /   /   /   /   /   /   /
#          61  37  19   7   1---2  11  26  47  74
#            \   \   \   \         /   /   /   /
#            62  38  20   8---9--10  25  46  73
#              \   \   \             /   /   /
#              63  39  21--22--23--24  45  72
#                \   \                 /   /
#                64  40--41--42--43--44  71
#                  \                     /
#                  65--66--67--68--69--70 '''

def hexagon_lattice_generator(yscale=1.7320508075688772935274):
  '''
  Generate points of a hexagonal lattice in a spiral order, similar to
  the figure above (but starting with 3).
  The parameter yscale can be set to:
  - sqrt(3): the default, which gives a regular hexagon lattice.
  - 1 for an all-integer algorithm, with any required scaling
    done outside this function.
  To scale to one point per unit area, use the factor 2/(1+sqrt(3)).
  See function below for verification.  Typical use:
  import numpy as np
  for xy in hexagon_lattice_generator():
    xy_scaled=0.535898384862245*np.array(xy) # factor is 2/(1+sqrt(3))
    ...
    if ...: break
  Another scaling which may be useful is to make the edge lengths 1:
  for xy in hexagon_lattice_generator():
    xy_scaled=0.5*np.array(xy)
    ...
    if ...: break
  '''
  cell=( # points 3,4,5,6,1,2,3 in A028896 picture; y value is row index, not real y coordinate
     (2, 0), (1, 1), (-1, 1), (-2, 0), (-1,-1), ( 1,-1), (2, 0),)
  yield (0,0)
  r=0
  while True:
    r+=1
    # code the small-r cases explicitly to avoid a loop...
    if r==1:
      for x,y in cell[:-1]:
        yield (x,yscale*y)
    elif r==2:
      for p0,p1 in zip(cell[:-1],cell[1:]):
        x0,y0=2*p0[0],2*p0[1]
        x1,y1=2*p1[0],2*p1[1]
        yield (x0,yscale*y0)
        yield ((x0+x1)//2,yscale*((y0+y1)//2))
    elif r==3:
      for p0,p1 in zip(cell[:-1],cell[1:]):
        x0,y0=3*p0[0],3*p0[1]
        x1,y1=3*p1[0],3*p1[1]
        yield (x0,yscale*y0)
        yield ((2*x0+x1)//3,yscale*((2*y0+y1)//3))
        yield ((x0+2*x1)//3,yscale*((y0+2*y1)//3))
    else: # r>=4, general case
      for p0,p1 in zip(cell[:-1],cell[1:]):
        x0,y0=r*p0[0],r*p0[1]
        x1,y1=r*p1[0],r*p1[1]
        yield (x0,yscale*y0)
        for i in range(1,r):
          j=r-i
          yield ((j*x0+i*x1)//r,yscale*((j*y0+i*y1)//r))

if __name__=='__main__':

  def test_00():
    i=0
    for xy in hexagon_lattice_generator():
      i+=1
      if xy is None or xy[0]>6: break
      print(f'{xy[0]}\t{xy[1]}')

  def fig_timestamp(fig,fontsize=6,color='black',alpha=0.7,rotation=0,prespace='  ',author='Keith Briggs'):
    from time import time,strptime,strftime,localtime
    date=strftime('%Y-%m-%d %H:%M',localtime())
    fig.text( # position text relative to Figure
      0.01,0.005,prespace+f'{author} {date}',
      ha='left',va='bottom',fontsize=fontsize,color=color,
      rotation=rotation,
      transform=fig.transFigure,alpha=alpha)

  def plot(fnbase='hexagon_lattice_generator_02'):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6, 'figure.autolayout': True})
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot()
    ax.grid(color='gray',lw=0.5,alpha=0.5)
    last_xy=None
    for i,xy in enumerate(hexagon_lattice_generator()):
      if xy[0]>10: break
      ax.plot(*xy,lw=0,marker='o',color='red',markersize=4)
      # shift labels slightly to avoid arrows...
      danx,dany=0.1,0.15
      if xy[0]*xy[1]<0: danx=0.35    # arrows on TL and BR sloping edges
      if -2.5<xy[1]<0.0 and xy[0]>0: # arrows between hexagons
        danx,dany=0.2,-0.2
      ax.annotate(f'{i}',(xy[0]+danx,xy[1]+dany),color='blue',fontsize=6)
      if last_xy:
        dx,dy=xy[0]-last_xy[0],xy[1]-last_xy[1]
        ax.arrow(*last_xy,0.9*dx,0.9*dy,color='black',head_width=0.1,length_includes_head=True)
      last_xy=(xy[0],xy[1])
    ax.set_aspect('equal')
    fig.tight_layout()
    fig_timestamp(fig,rotation=0,fontsize=4,author='Keith Briggs')
    if fnbase:
      fig.savefig(f'{fnbase}.pdf')
      print(f'evince {fnbase}.pdf &')
    else:
      plt.show()

  def points_per_unit_area(r_max=1000.0,scale=0.53589838486224541294510):
    from math import hypot,pi
    k=0
    for x,y in hexagon_lattice_generator():
      r=scale*hypot(x,y)
      if r<r_max: k+=1
      if r>2*r_max: break # no more points in circle
    a=pi*r_max**2
    ppua=k/a
    print(f'{k} points in area {a:.6f} => {ppua:.6f} points per unit area after scaling by 2/(1+sqrt(3))≈0.5358983848...')

  from sys import argv
  if len(argv)<=1:
    test_00()
  elif 'plot' in argv[1] or 'pdf' in argv:
    plot()
  elif 'ppua' in argv:
    points_per_unit_area()
