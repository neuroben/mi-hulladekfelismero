# Fejlesztői napló

## Miről szól ez az alkalmazás

Ezt az alkalmazást azért készítettem el, hogy képfelismeréssel meg tudjam mondani egy hulladékról készült fotó alapján, hogy az üveg, fém, papír vagy műanyag kategóriába tartozik-e. A projektet PyTorch alapokra építettem, a felhasználói felülethez Gradio-t használtam, a gyorsabb tréninghez pedig Modalon futtattam GPU-s tanítást.

## Hogyan építettem fel a projektet

A fejlesztést egy egyszerűbb, lokálisan futó osztályozó projekttel kezdtem, majd fokozatosan átrendeztem egy tisztább, professzionálisabb struktúrába. Különválasztottam a közös logikát és a futtatható belépési pontokat. A közös részeket egy külön Python package-be szerveztem, így a modellépítés, az adatbetöltés, az inferencia, az értékelés és a GUI nem szanaszét elhelyezett szkriptekből áll, hanem egységesen karbantartható modulokból.

Az alkalmazás jelenlegi felépítésében külön kezeltem:

- az adatbetöltést és augmentációt,
- a modell definícióját,
- a tanítási ciklust,
- a kiértékelést,
- az egyképes predikciót,
- a Gradio alapú webes felületet,
- valamint a Modalon futó remote tréning workflow-t.

Emellett megtartottam a régi parancsok kompatibilitását is, hogy a projekt használata ne törjön meg a refactor után.

## Milyen adatokkal dolgoztam

Az alap projektet egy kisebb hulladékos képhalmazra építettem, majd ezt kibővítettem a HOWA adattal. A HOWA datasetben az annotációk JSON fájlokban voltak megadva, ezért a képek egyszerű bemásolása helyett annotáció alapján cropolt objektumképeket készítettem. Ez fontos döntés volt, mert a teljes képeken a hulladékobjektumok sokszor túl kicsik voltak, classifierhez viszont a kivágott objektum sokkal hasznosabb tanító minta lett.

Az osztályaim a teljes projektben ezek maradtak:

- `glass`
- `metal`
- `paper`
- `plastic`

A HOWA esetében a `carton` kategóriát a `paper` osztályhoz képeztem le, mert a jelenlegi modell négyosztályos anyagfelismerésre készült.

A HOWA bővítés után a dataset összesen `8929` képet tartalmazott, ebből:

- `train`: `6249`
- `val`: `1685`
- `test`: `995`

Az osztályok nagyjából kiegyensúlyozottak lettek, ami különösen fontos volt ahhoz, hogy a modell ne csak egy-két gyakori kategóriát tanuljon meg jól.

A projekt későbbi szakaszában még tovább bővítettem a képhalmazt a Kaggle Garbage Classification V2 forrásból is. Ebből a `glass`, `metal`, `paper` és `plastic` almappákat vettem át, majd reprodukálható, fix seed alapú split szerint szétosztottam a meglévő `train`, `val` és `test` struktúrába. Ezzel a teljes, véglegesített dataset `14528` képre nőtt.

## Hogyan tanítottam a modellt

Az alkalmazás alapja egy előtanított `ResNet18` modell lett. A tanítást transfer learning megközelítéssel oldottam meg, vagyis nem nulláról tanítottam egy teljes hálózatot, hanem egy már előtanított képfelismerő modellt finomhangoltam a saját négy hulladékosztályomra.

Lokálisan is futtattam rövidebb próbákat, de amikor már nagyobb dataset állt rendelkezésre, regisztráltam a Modalra, beállítottam az API tokent, feltöltöttem a datasetet egy tartós Volume-ba, és ott GPU-n futtattam a tréninget. Ez azért volt hasznos, mert a CPU-s tanítás a saját gépemen érezhetően lassabb volt, míg Modalon egy L4 GPU-val sokkal gyorsabban végig tudtam menni ugyanazon a folyamaton. A későbbi szakaszban a még erősebb B200 GPU-t is kipróbáltam ugyanennek a projektnek a tanítására.

A korábbi, HOWA-bővítés utáni fontos Modalos futásom:

- Modal L4 GPU
- `6` epoch
- `batch_size = 64`
- `learning_rate = 0.001`

A legfrissebb, kibővített képhalmazon futtatott erősebb GPU-s köröm pedig ez lett:

- Modal B200 GPU
- `6` epoch
- `batch_size = 128`
- `learning_rate = 0.001`
- run név: `kaggle_howa_b200_run2`

## Milyen eredményt értem el

A HOWA-bővítés utáni Modalos tanításnál ezeket az eredményeket értem el:

- legjobb validációs pontosság: `0.8059`
- tesztpontosság: `0.8161`

Az osztályonkénti eredmények közül a `paper` kategória teljesített a legerősebben, míg az `glass` és `plastic` között még most is látszik némi átfedés, ami a valós tárgyformák és az áttetsző anyagok miatt érthető. A modell tehát már használható, de még nem tekintem késznek vagy ipari szintűnek.

A Kaggle Garbage Classification V2 képeinek hozzáadása után a dataset még tovább bővült, és erre a nagyobb, `14528` képes adathalmazra már lefuttattam egy új Modalos tanítást B200 GPU-n is. Ennél a legfrissebb körnél ezeket az eredményeket értem el:

- legjobb validációs pontosság: `0.8161`
- tesztpontosság: `0.8069`

## A két modell összehasonlítása

A két legerősebb futásom végül így alakult:

- `howa_modal_run1`:
  HOWA-val bővített dataset (`8929` kép), Modal `L4`, `batch_size = 64`, legjobb `val_acc = 0.8059`, `test_acc = 0.8161`, teszthalmaz = `995` kép
- `kaggle_howa_b200_run2`:
  HOWA + Kaggle V2 alapján kibővített dataset (`14528` kép), Modal `B200`, `batch_size = 128`, legjobb `val_acc = 0.8161`, `test_acc = 0.8069`, teszthalmaz = `1838` kép

Ebből az látszik, hogy a B200-as futás már a nagyobb, nehezebb és változatosabb adathalmazon ment, ezért nem teljesen egy az egyben összehasonlítható a korábbi L4-es körrel. A gyorsító elsősorban a sebességen segít, de a végeredményt ugyanúgy erősen befolyásolja az adatösszetétel, a split, a batch méret és a hiperparaméterezés. A mostani mérésem alapján az L4-es modell picit magasabb tesztpontosságot adott, viszont ezt egy kisebb, `995` képes teszthalmazon mértem. A B200-as modell ezzel szemben majdnem ugyanilyen szintet hozott egy jóval nagyobb, `1838` képes teszthalmazon, ami szerintem kifejezetten biztató eredmény.

Osztályszinten is érdekes a különbség:

- az L4-es modell `metal` osztályban erősebb maradt, és összesített tesztpontosságban is enyhén előrébb végzett;
- a B200-as, kibővített képes futásnál a `paper` továbbra is nagyon stabil maradt (`f1 = 0.89`);
- a B200-as modell `glass` osztályban kissé jobb F1-t adott (`0.79` az `0.78` helyett), miközben `plastic` és `metal` oldalon valamivel vegyesebb lett a kép;
- ebből arra következtetek, hogy a nagyobb dataset és az erősebb GPU jó irány, de a kibővített adathalmazhoz még érdemes külön hiperparaméter-hangolást is végezni.

## Milyen GUI készült hozzá

Az alkalmazáshoz készítettem egy egyszerű Gradio alapú webes felületet. A felhasználó ezen a felületen feltölt egy képet, az alkalmazás pedig visszaadja:

- a prediktált osztályt,
- valamint az egyes osztályokhoz tartozó valószínűségeket.

Ez a GUI gyors demóra és ellenőrzésre teljesen megfelelő, mert külön telepített frontend nélkül, pár parancsból helyben elindítható.

A projekt legvégén egy nagyobb refaktort is elvégeztem. Ekkor kiszerveztem a közös logikát egy rendezettebb Python package struktúrába, külön `scripts/` belépési pontokat alakítottam ki, és megtartottam a régi parancsok kompatibilitását is. Ezzel a projekt sokkal tisztább és professzionálisabb formát kapott, miközben a használhatósága megmaradt.

## A jelenlegi GUI limitációi

A mostani felületnek több tudatosan vállalt korlátja van:

- csak `4` osztályt kezel;
- egyszerre egy képet értékel ki;
- mindig választ egy kategóriát, akkor is, ha a kép bizonytalan vagy nem tartozik egyik osztályba sem;
- nem jelöl ki objektumot a képen;
- nem detektál több hulladékot ugyanazon a fotón;
- inkább demonstrációs és fejlesztői felület, mint végleges termék GUI.

## Mivel lehet továbbfejleszteni

A projektet több irányban is tovább lehet vinni.

Az első és legfontosabb lépés további datasetek bevonása lenne, mert a modell jelenlegi minőségét még erősen befolyásolja, hogy milyen háttérrel, fényviszonnyal és tárgytípussal találkozott tanítás közben. Emellett érdemes lenne kipróbálni erősebb backbone-okat, például modernebb EfficientNet vagy ConvNeXt jellegű architektúrákat.

Szakmailag izgalmas következő lépés lenne:

- a teljes backbone finomhangolása több körben,
- külön `unknown` vagy visszautasító osztály bevezetése,
- több valós környezetből származó adat bevonása,
- object detection alapú megközelítés,
- mobilbarát vagy szebb, részletesebb GUI kialakítása,
- predikciós küszöbök és bizonytalanságkezelés bevezetése.

## Összegzés

Ezzel a projekttel egy működő, négyosztályos hulladékfelismerő alkalmazást építettem fel, amely lokálisan és felhőben is futtatható. A fejlesztés során kialakítottam a dataset struktúrát, bővítettem az adatforrásokat, beemeltem a HOWA annotált adatait, hozzáadtam a Kaggle Garbage Classification V2 képeit, megoldottam a GPU-s tréninget Modalon, készítettem hozzá egy használható webes demófelületet, és a végén egy átfogóbb refaktorral professzionálisabb szerkezetbe rendeztem a teljes projektet.

A mostani állapotot egy stabil, jól bemutatható alapnak tartom, amire már érdemes további modellezési és termékesítési lépéseket építeni.
