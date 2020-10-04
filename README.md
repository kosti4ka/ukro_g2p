# ukro_g2p

## Installation

`pip install -U ukro-g2p`

## Example inference

```python
from ukro_g2p.predict import G2P

g2p = G2P('ukro-base-uncased')

#ARPABET format
g2p('фонетика')

#human readable format
g2p('фонетика', human_readable=True)
```

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bptBFKWtuBVVbAA_e_WB5tL-N4tJ4xyq#scrollTo=JGG5NcltvXTx?usp=sharing)

## Web app
https://ukro-g2p.herokuapp.com

Code for the web app: https://github.com/kosti4ka/ukro_g2p_demo

## Ukrainian phonology symbols

### Голосні
<table>
    <thead>
        <tr>
            <th>Ukrainian</th>
            <th>ARPABET-like</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[і]</td>
            <td>IY</td>
        </tr>
        <tr>
            <td>[и]</td>
            <td>IH</td>
        </tr>
        <tr>
            <td>[е]</td>
            <td>EH</td>
        </tr>
        <tr>
            <td>[у]</td>
            <td>UH</td>
        </tr>
        <tr>
            <td>[о]</td>
            <td>AO</td>
        </tr>
        <tr>
            <td>[а]</td>
            <td>AA</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th colspan="2">Наближення</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[е<sup>и</sup>]</td>
            <td>EIH</td>
        </tr>
        <tr>
            <td>[е<sup>і</sup>]</td>
            <td>EIY</td>
        </tr>
        <tr>
            <td>[и<sup>е</sup>]</td>
            <td>IHE</td>
        </tr>
        <tr>
            <td>[о<sup>у</sup>]</td>
            <td>AOU</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th colspan="2">Наголос</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[ ́ ]</td>
            <td>1</td>
        </tr>
    </tbody>
</table>

### Приголосні
<table>
    <thead>
        <tr>
            <th>Ukrainian</th>
            <th>ARPABET-like</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[б]</td>
            <td>B</td>
        </tr>
        <tr>
            <td>[в]</td>
            <td>V</td>
        </tr>
        <tr>
            <td>[г]</td>
            <td>H</td>
        </tr>
        <tr>
            <td>[д]</td>
            <td>D</td>
        </tr>
        <tr>
            <td>[дж]</td>
            <td>JH</td>
        </tr>
        <tr>
            <td>[дз]</td>
            <td>DZ</td>
        </tr>
        <tr>
            <td>[ж]</td>
            <td>ZH</td>
        </tr>
        <tr>
            <td>[з]</td>
            <td>Z</td>
        </tr>
        <tr>
            <td>[й]</td>
            <td>Y</td>
        </tr>
        <tr>
            <td>[к]</td>
            <td>K</td>
        </tr>
        <tr>
            <td>[л]</td>
            <td>L</td>
        </tr>
        <tr>
            <td>[м]</td>
            <td>M</td>
        </tr>
        <tr>
            <td>[н]</td>
            <td>N</td>
        </tr>
        <tr>
            <td>[п]</td>
            <td>P</td>
        </tr>
        <tr>
            <td>[р]</td>
            <td>R</td>
        </tr>
        <tr>
            <td>[с]</td>
            <td>S</td>
        </tr>
        <tr>
            <td>[т]</td>
            <td>T</td>
        </tr>
        <tr>
            <td>[х]</td>
            <td>X</td>
        </tr>
        <tr>
            <td>[ц]</td>
            <td>TS</td>
        </tr>
        <tr>
            <td>[ч]</td>
            <td>CH</td>
        </tr>
        <tr>
            <td>[ш]</td>
            <td>SH</td>
        </tr>
        <tr>
            <td>[ґ]</td>
            <td>G</td>
        </tr>
        <tr>
            <td>[ў]</td>
            <td>WH</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th colspan="2">М'які</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[д’]</td>
            <td>DJ</td>
        </tr>
        <tr>
            <td>[дз’]</td>
            <td>DZJ</td>
        </tr>
        <tr>
            <td>[з’]</td>
            <td>ZJ</td>
        </tr>
        <tr>
            <td>[л’]</td>
            <td>LJ</td>
        </tr>
        <tr>
            <td>[н’]</td>
            <td>NJ</td>
        </tr>
        <tr>
            <td>[р’]</td>
            <td>RJ</td>
        </tr>
        <tr>
            <td>[с’]</td>
            <td>SJ</td>
        </tr>
        <tr>
            <td>[т’]</td>
            <td>TJ</td>
        </tr>
        <tr>
            <td>[ц’]</td>
            <td>TSJ</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th colspan="2">Пом'якшення</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[`]</td>
            <td>2</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th colspan="2">Подовження</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[:]</td>
            <td>3</td>
        </tr>
    </tbody>
</table>
