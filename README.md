# Projeto - Classificador de Imagens Histopatológicas de Câncer Pulmonar

Segundo projeto apresentado na displina de Inteligência Computacional em Saúde ministrada pelo professor [Andre Georghton Cardoso Pacheco](https://github.com/paaatcha). Os alunos envolvidos no desenvolvimento foram:
<ul>
    <li>Luiz Carlos Cosmi Filho</li>
        <a href="http://lattes.cnpq.br/7512442154273401">
            <img src="https://img.shields.io/badge/Lattes-0A0A0A?style=for-the-badge" alt="Dev"/>
        </a>
        <a href="mailto:luizcarloscosmifilho@gmail.com">
            <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Mail"/>
        </a>
        <a href="https://www.linkedin.com/in/luizcarloscf/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"/>
        </a>
        <a href="https://github.com/luizcarloscf">
            <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="github"/>
        </a>
    <li>Mateus Sobrinho Menines</li>
        <a href="http://lattes.cnpq.br/0283141894444882">
            <img src="https://img.shields.io/badge/Lattes-0A0A0A?style=for-the-badge" alt="Dev"/>
        </a>
        <a href="mailto:mateus.sobrinho09@gmail.com">
            <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Mail"/>
        </a>
        <a href="https://www.linkedin.com/in/mateus-sobrinho-868147256/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"/>
        </a>
        <a href="https://github.com/MateusSMenines">
            <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="github"/>
        </a>
</ul>

## Sumário

- [Base de dados](#base-de-dados)
- [Licença](#licença)

## Base de dados

A base de dados utilizada neste projeto corresponde a um conjunto de dados de imagens histopatológicas de câncer de pulmão e cólon chamado [LC25000](https://arxiv.org/abs/1912.12142v1). Contém 25.000 imagens histopatológicas (RGB) com resolução de 728x728 de tecidos do pulmão e cólon, distribuídas em 5 classes, com cada classe tendo 5.000 imagens: (i) Adenocarcinoma de Cólon; (ii) Tecido Benigno do Cólon; (iii) Adenocarcinoma Pulmonar; (iv) Tecido Pulmonar Benigno; (v) Carcinoma de Células Escamosas Pulmonar. É importante mencionar que esses tecidos foram preparados utilizando uma técnica de coloração com hematoxilina e eosina. Nesse trabalho, serão utilizadas apenas as imagens relativas a **tecidos pulmonares**.

|Classe                                   |  Exemplo                           |
|:---------------------------------------:|:----------------------------------:|
| Adenocarcinoma Pulmonar                 | <img src="./data/examples/lungaca1.jpeg" alt="drawing" width="200"/> | 
| Tecido Pulmonar Benigno                 | <img src="./data/examples/lungn1.jpeg" alt="drawing" width="200"/>   |
| Carcinoma de Células Escamosas Pulmonar | <img src="./data/examples/lungscc1.jpeg" alt="drawing" width="200"/> |


## Licença

Este projeto é licenciado sob os termos da [licença MIT](./LICENSE) e está disponível gratuitamente.