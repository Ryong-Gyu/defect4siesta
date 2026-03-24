# defect4siesta workflow

## Single-run flow

1. **Structure generation**  
   `python main.py <ground.fdf> <excited.fdf> --mode generate`  
   (or `--mode generate-neb` when cell interpolation is needed)
2. **Calculation setup + submission**  
   The generate modes automatically call directory setup and `sbatch` submission.
3. **CC fitting/plot export**  
   `python main.py <ground.fdf> <excited.fdf> --mode cc`
4. **(Optional) NMP capture-time calculation**  
   Add `--run-nmp --nmp-output nmp.txt` to the CC command.

## Standalone NMP run

```bash
python nmp.py --dQ <amu^1/2·Å> --dE <eV> --wi <eV> --wf <eV>
```
