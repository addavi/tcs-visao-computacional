# TCS - Visão Computacional
### Resultado

- **Objetos detectados (classe `car`, YOLOv8n): 17**
- **Confiança média (YOLOv8n): 0.845**
- **Contagem total de objetos: 22 carros**
- Imagem anotada: `q1_yolo/saida/estacionamento_anotado_v8n.jpg`


### Execução

```bash
cd q1_yolo
python detectar.py
```

O script:
1. Carrega o modelo YOLO pré-treinado (baixa os pesos automaticamente na primeira execução, ex. `yolov8n.pt`);
2. Detecta objetos da classe `car` na imagem `imagens/estacionamento.jpg`;
3. Imprime no terminal a contagem total e a confiança média das detecções;
4. Salva a imagem anotada com as caixas de detecção em `saida/estacionamento_anotado.jpg`.

Para reproduzir a comparação entre modelos (YOLOv8n vs YOLOv8s) mencionada na seção de validação, basta trocar a linha:
```python
modelo = YOLO("yolov8n.pt")
```
por:
```python
modelo = YOLO("yolov8s.pt")
```
e rodar novamente.

### Justificativas

**Qual modelo/versão usei e por quê?**

Utilizei o **YOLOv8n (nano)** da Ultralytics como modelo principal, com pesos pré-treinados no dataset COCO. Fiz essa escolha por três motivos:

1. A classe `car` já faz parte das 80 classes do COCO — não há necessidade de treinar ou fazer fine-tuning.
2. A variante nano é a mais leve da família, então ocorre o processamento mais rápido.
3. A API da Ultralytics entrega o pós-processamento pronto (Supressão não máxima:cria varias caixas candidatas com diferentes pontuações e seleciona a melhor, filtro de confiança selecionando a melhor pontuada e a criação das caixas delimitadoras), reduzindo código desnecessário.

Porém, a validação manual (22 carros reais vs. 17 detectados) revelou que o modelo nano perde carros em cantos da imagem e em situações de oclusão parcial. Para confirmar a causa, testei o **YOLOv8s**, que subiu a contagem para 21/22 (recall de ~95%), com uma queda pequena e esperada na confiança média (0.824 vs 0.845, já que o modelo passa a detectar casos mais difíceis, com menor certeza). 

**Como lidei com detecções duplicadas ou de baixa confiança?**

- **Baixa confiança:** apliquei a variavel limiar `conf=0.4`. Com a confiança média das detecções em 0.845 (nano) e 0.824 (small), ambas bem acima do limiar, a margem confirma que o corte não estava escondendo falsos positivos nem sustentando artificialmente a contagem.
- **Duplicatas:** o pipeline do YOLO aplica *Non-Maximum Suppression* (NMS), controlado por `iou=0.5`. Quando duas caixas da mesma classe se sobrepõem acima desse IOU, mantém-se apenas a de maior confiança. Um valor mais restritivo (ex. `iou=0.3`) tornaria a supressão mais forte, mas no meu caso não houve detecções duplicadas (0 falsos positivos), então o valor padrão de 0.5 se mostrou adequado, um IOU menor arriscaria anular carros verdadeiramente enfileirados e próximos, que é exatamente o padrão de erro observado nesse dataset (carros próximos entre si).

**Qual foi a confiança média das detecções contadas?**

**0.845** com YOLOv8n (17 detecções) e **0.824** com YOLOv8s (21 detecções), médias aritméticas do score de confiança das caixas da classe `car` que passaram pelo filtro `conf=0.4`.

---

## Questão 2 — Segmentação com visão tradicional

### Execução

```bash
cd q2_segmentacao
python segmentar.py
```

### Imagem utilizada

`skimage.data.coins()` — Imagem pública embutida no scikit-image, com 24 moedas reais sobre fundo escuro.

### Abordagem

1. Suavização leve (filtro gaussiano, `sigma=1`) para reduzir ruído antes da limiarização.
2. **Limiarização global de Otsu** para separar moedas (objeto) do fundo.
3. Limpeza morfológica (`remove_small_objects`, `remove_small_holes`) para eliminar ruído residual.
4. **Transformada de distância euclidiana** + detecção de máximos locais para gerar marcadores individuais por moeda.
5. **Watershed** para separar moedas que estão encostadas/próximas, usando os marcadores como sementes.

### Resultado

- **Contagem: 25 objetos**
- Imagens salvas em `q2_segmentacao/saida/` (segmentação binária e Watershed colorido)

### Validação e caso de falha observado

O valor real de moedas na imagem é **24**. O método contou **25**, ou seja, um falso positivo identificado na prática.

Analisando a imagem de segmentação binária, o erro está concentrado na **faixa superior da imagem**: uma região do fundo, próxima ao topo, foi marcada incorretamente como um objeto pelo Otsu, o que formou uma mancha irregular que o Watershed depois marcou como uma "moeda" extra (é visível como uma área avermelhada cobrindo parte da primeira fileira no resultado final).

A causa provavel é **iluminação não uniforme**: o fundo da imagem `coins()` não é perfeitamente padronizado, a região superior tem um tom levemente mais claro que o restante do fundo. Como a limiarização de Otsu calcula **um único limiar global** para a imagem inteira, esse limiar funciona bem nas fileiras inferiores (mais fundo e mais escuro), mas na região superior o fundo mais claro ultrapassa o limiar e é falsamente incluído como "objeto".

### Justificativas

**Por que escolhi esse método de limiarização/segmentação?**

Optei pelo Otsu por ser um método automático (não exige ajuste manual de um valor fixo) e ideal para imagens com dois grupos de intensidade diferentes, como moedas claras sobre fundo escuro. Combinei isso com Watershed porque a limiarização sozinha, mesmo perfeita, não separa objetos que se tocam fisicamente (ela produz uma única "blob" conectada); o Watershed, usando a transformada de distância, encontra os "picos" (centros prováveis de cada moeda) e separa a região em circulos individuais, mesmo quando duas moedas estão encostadas.

**O que aconteceria se a iluminação da cena mudasse?**

O resultado mudaria de forma proporcional à não uniformidade da iluminação, igual aconteceu nesse teste. Como o Otsu assume um limiar único e global, qualquer gradiente de luz (sombra de um lado, reflexo de outro, iluminação mais intensa no centro) faz com que esse limiar deixe de ser ideal para toda a imagem simultaneamente. Regiões mais claras do fundo passam a ser confundidas com objetos (como ocorreu na faixa superior desta imagem), e regiões mais escuras dos próprios objetos poderiam deixar de ser detectadas. Em um cenário com iluminação mais desigual que o desta imagem de teste, o erro tenderia a ser ainda maior.

**Cite um caso em que seu método falharia**

Este próprio experimento é o caso: a segmentação contou **25 moedas em vez de 24** porque uma faixa do fundo, na parte superior da imagem, tem um tom mais claro que o restante e foi segmentada como um objeto adicional pelo limiar global do Otsu. Isso ilustra uma limitação do método: ele não diferencia "objeto real" de "qualquer região que ultrapasse o brilho do limiar", então qualquer deformação na iluminação do fundo, o método tende a falhar. Uma melhora possível seria usar limiarização adaptativa/local (`threshold_local`) em vez de global, calculando um limiar diferente por cada região da imagem.


## Questão 3 — Tanque com espuma

### Câmera

Colocaria uma câmera fixa do lado do tanque, olhando pra escada inteira, do primeiro degrau até o topo. O importante é ela nunca se mexer depois de instalada, porque vou usar a escada como "régua" pra medir a altura. Colocaria também uma luz fixa perto da câmera, pra não depender da luz do ambiente que muda o dia todo. Fazer no mesmo estilo da questão 2: limiar adaptativo pra achar onde a espuma toca a escada, e variação de brilho pra estimar densidade.

### Altura

Como os degraus são igualmente espaçados, dá pra usar eles como referência de medida — é só saber a distância real entre um degrau e outro (isso mediria uma vez só, na instalação) já que a câmera é fixa e a referência (escada) não muda de lugar..

Na prática:
1. Marcar a posição de cada degrau na imagem, uma única vez.
2. O sistema identifica onde a espuma "encosta" na escada (a espuma tem uma textura/brilho diferente do degrau seco, assim da pra interpretar isso).
3. Sabendo em qual degrau (ou fração de degrau) a espuma está batendo, convertemos isso pra altura real.


### Densidade

Como diz no enunciado: bolha grande e definida = menos densa, superfície lisa = mais densa. Isso dá pra pegar olhando a variação de brilho entre pixels vizinhos numa região pequena da imagem (por exemplo, um quadradinho de 20x20 pixels na superfície da espuma):

- Se essa variação for alta -> tem bolha aparecendo, contraste entre luz e sombra -> espuma menos densa.
- Se for baixa -> superfície mais uniforme -> espuma mais densa.

### O que precisaria de dados

- Um operador batendo o olho em algumas imagens e classificando a densidade numa escala simples (exemplo de 1 a 5), só pra eu calibrar o que é "alto" ou "baixo" na variação de brilho.
- Medir a distância real entre os degraus, uma vez.

### O que pode dar errado e como resolver

**Luz mudando ao longo do dia** — afeta tanto achar a linha da espuma quanto medir a densidade. Resolveria com luz de led fixa perto da câmera e usando limiar adaptativo em vez de um valor fixo. (Igual a solução pensada para Q2)

**Reflexo na espuma** — pode criar um brilho forte que atrapalha a leitura. Um filtro polarizador na lente poderia resolver uma boa parte, sem precisar mexer no código.

**Espuma balançando/mudando rápido** — uma foto isolada pode não representar bem o momento real. Em vez de olhar um frame só, tiraria a média das últimas leituras (uns 10 segundos de vídeo). Assim, aumentaria a precisão do programa.