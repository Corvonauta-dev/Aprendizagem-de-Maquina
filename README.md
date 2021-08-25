# Aprendizagem-de-Maquina
Aplicação de algoritmos de aprendizagem de máquina no game Sapace Invaders de ATARI, para fins de estudos e agregação de conhecimento.

Foi selecionado uma aplicação em jogos, mais especificamente ao famoso jogo “Space Invaders” de 1978, para a plataforma de vídeo games Atari, aplicado ao jogo foram utilizados dois algoritmos de aprendizagem por reforço, Deep Q-Networks e Cross-Entropy Method, a fim de otimizar a quantidade de pontos obtidos em uma partida com 3 tentativas.


Para a compilação e execução do programa é preciso:
- Executar o comando
  pip install tensorflow==2.5.0 gym keras-rl2 gym[atari]
- Instalar o emulador Stella
- Colar a dll ale_c.dll na pasta C:\Python39\Lib\site-packages\atari_py\ale_interface
- Executar o comando
  pip install gym[all]
