# Keith Briggs 2024-01-11
'''
For codes see https'://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
self-test: python3 color_text_00.py
install: cp color_text_00.py ~/.local/lib/python3.8/site-packages/color_text.py
normal use:
  from color_text import Color_text
  color_text=Color_text()
  print(color_text.on_white(color_text.bred('This is on white bold red')))
'''

_color_table={
  # Reset...
  'Color_Off': '\033[0m',       # Text Reset
  # Regular Colors...
  'Black':  '\033[0;30m',       # Black
  'Red':    '\033[0;31m',       # Red
  'Green':  '\033[0;32m',       # Green
  'Yellow': '\033[0;33m',       # Yellow
  'Blue':   '\033[0;34m',       # Blue
  'Purple': '\033[0;35m',       # Purple
  'Cyan':   '\033[0;36m',       # Cyan
  'White':  '\033[0;37m',       # White
  # Bold...
  'BBlack':  '\033[1;30m',      # Black
  'BRed':    '\033[1;31m',      # Red
  'BGreen':  '\033[1;32m',      # Green
  'BYellow': '\033[1;33m',      # Yellow
  'BBlue':   '\033[1;34m',      # Blue
  'BPurple': '\033[1;35m',      # Purple
  'BCyan':   '\033[1;36m',      # Cyan
  'BWhite':  '\033[1;37m',      # White
  # Underline...
  'UBlack':  '\033[4;30m',      # Black
  'URed':    '\033[4;31m',      # Red
  'UGreen':  '\033[4;32m',      # Green
  'UYellow': '\033[4;33m',      # Yellow
  'UBlue':   '\033[4;34m',      # Blue
  'UPurple': '\033[4;35m',      # Purple
  'UCyan':   '\033[4;36m',      # Cyan
  'UWhite':  '\033[4;37m',      # White
  # Background...
  'On_Black':  '\033[40m',      # Black
  'On_Red':    '\033[41m',      # Red
  'On_Green':  '\033[42m',      # Green
  'On_Yellow': '\033[43m',      # Yellow
  'On_Blue':   '\033[44m',      # Blue
  'On_Purple': '\033[45m',      # Purple
  'On_Cyan':   '\033[46m',      # Cyan
  'On_White':  '\033[47m',      # White
  # High Intensity...
  'IBlack':  '\033[0;90m',      # Black
  'IRed':    '\033[0;91m',      # Red
  'IGreen':  '\033[0;92m',      # Green
  'IYellow': '\033[0;93m',      # Yellow
  'IBlue':   '\033[0;94m',      # Blue
  'IPurple': '\033[0;95m',      # Purple
  'ICyan':   '\033[0;96m',      # Cyan
  'IWhite':  '\033[0;97m',      # White
  # Bold High Intensity...
  'BIBlack':  '\033[1;90m',     # Black
  'BIRed':    '\033[1;91m',     # Red
  'BIGreen':  '\033[1;92m',     # Green
  'BIYellow': '\033[1;93m',     # Yellow
  'BIBlue':   '\033[1;94m',     # Blue
  'BIPurple': '\033[1;95m',     # Purple
  'BICyan':   '\033[1;96m',     # Cyan
  'BIWhite':  '\033[1;97m',     # White
  # High Intensity backgrounds...
  'On_IBlack':  '\033[0;100m',  # Black
  'On_IRed':    '\033[0;101m',  # Red
  'On_IGreen':  '\033[0;102m',  # Green
  'On_IYellow': '\033[0;103m',  # Yellow
  'On_IBlue':   '\033[0;104m',  # Blue
  'On_IPurple': '\033[0;105m',  # Purple
  'On_ICyan':   '\033[0;106m',  # Cyan
  'On_IWhite':  '\033[0;107m',  # White
}

class Color_text:
  def __init__(s):
    for color,code in _color_table.items():
      q=lambda x,code=code: f'{code}{x}\033[0m' # closure
      s.__dict__[color]=q
      s.__dict__[color.lower()]=q # also lower-case name, for convenience

def _test_color_text():
  color_text=Color_text()
  print(color_text.Red('This is Red'))
  print(color_text.IRed('This is IRed'))
  print(color_text.BIRed('This is BIRed'))
  print(color_text.red('This is red'))
  print(color_text.ired('This is ired'))
  print(color_text.bired('This is bired'))
  print(color_text.biyellow('This is biyellow'))
  print(color_text.cyan('This is cyan'))
  print(color_text.on_white('This is on white'))
  print(color_text.cyan(color_text.on_white('This is cyan on white')))
  print(color_text.on_white(color_text.cyan('This is on white cyan')))
  print(color_text.on_white(color_text.bred('This is on white bold red')))

if __name__=='__main__':
  _test_color_text()
