"""
fig_timestamp_01.py
----------------
A module with a function that adds a timestamp and the author's name to matplotlib figures.

Author: Keith Briggs [2020-02-12].
"""

from time import strftime,localtime

def fig_timestamp(fig,fontsize=6,color='black',alpha=0.7,rotation=0,prespace='  ',author='Keith Briggs'):
  """
  Adds a timestamp and author information to a Matplotlib figure.

  Parameters
  ----------
  fig : matplotlib.figure.Figure
    The figure object to which the timestamp will be added.
  fontsize : int, optional
    Font size of the timestamp text. Default is 6.
  color : str, optional
    Color of the timestamp text. Default is 'black'.
  alpha : float, optional
    Opacity of the timestamp text (0.0 transparent through 1.0 opaque). Default is 0.7.
  rotation : float, optional
    Rotation angle of the timestamp text in degrees. Default is 0.
  prespace : str, optional
    String to prepend before the author and date. Default is two spaces.
  author : str, optional
    Name to display as the author. Default is 'Keith Briggs'.

  Notes
  -----
  The timestamp is placed at the bottom left of the figure using figure-relative coordinates.
  Adapted from  https://riptutorial.com/matplotlib/example/16030/coordinate-systems-and-text.
  """
  date=strftime('%Y-%m-%d %H:%M',localtime())
  fig.text( # position text relative to Figure
    0.01,0.005,prespace+f'{author} {date}',
    ha='left',va='bottom',fontsize=fontsize,color=color,
    rotation=rotation,
    transform=fig.transFigure,alpha=alpha)
