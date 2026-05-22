# Neural Chess Bot

Motor de ajedrez con red neuronal tipo AlphaZero. Implementa MCTS (Monte Carlo Tree Search) + ResNet entrenado por auto-juego (self-play).

## Arquitectura

```
Input (17x8x8) → ConvBlock → [ResBlock x10] → Policy Head (4672) + Value Head (1)
```

| Componente | Descripción |
|------------|-------------|
| `ChessEnv` | Estado del tablero como tensor 17x8x8 |
| `ChessResNet` | Red residual con policy + value heads |
| `MCTS` | Búsqueda Monte Carlo con selección UCB |
| `SelfPlay` | Genera partidas jugando contra sí mismo |
| `Trainer` | Entrena con pérdida combinada value + policy |

## Setup

```bash
pip install -r requirements.txt
python tests/test_game.py
python main.py
```

## Entrenamiento (Google Colab)

1. Abrir notebook en Colab con GPU
2. Clonar repo: `!git clone https://github.com/vanefalmiron/neural-chess-bot.git`
3. Instalar dependencias: `!pip install -r requirements.txt`
4. Ejecutar celda de entrenamiento con TensorBoard

Checkpoints se guardan automáticamente en Google Drive cada 5 iteraciones.

## Jugar contra la IA

```python
from src.game.chess_env import ChessEnv
from src.mcts.mcts import MCTS
from src.neural_net.resnet import ChessResNet

model = ChessResNet(num_blocks=5, channels=64)
# cargar checkpoint...

env = ChessEnv()
mcts = MCTS(model, num_simulations=100)
best_move = mcts.get_best_move(env, temperature=0)
```

## Roadmap

- [x] Lógica del juego (python-chess)
- [x] Red neuronal ResNet
- [x] MCTS básico
- [x] Self-play loop
- [x] Entrenamiento con TensorBoard
- [ ] UCI engine para jugar contra otros motores
- [ ] Web interface con tablero interactivo
- [ ] Pre-entrenamiento con partidas humanas (Lichess DB)

## Recursos

- [AlphaZero paper](https://arxiv.org/abs/1712.01815) (Silver et al., 2017)
- [python-chess docs](https://python-chess.readthedocs.io/)
- [Suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - referencia PyTorch

## Licencia

MIT
