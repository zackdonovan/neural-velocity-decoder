# Data attribution and scope

Churchland, Mark; Kaufman, Matthew (2022). MC_Maze: macaque primary motor and dorsal premotor cortex spiking activity during delayed reaching. Version 0.220113.0400. DANDI archive. https://doi.org/10.48324/dandi.000128/0.220113.0400 .

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ . Original authors do not endorse this analysis.

This package transforms the official training NWB into M1-unit spike-count histories and contemporaneous, event-aligned hand-velocity targets. Counts are binned, velocity samples averaged, and new source-specific models trained. The original recordings were collected by the credited dataset authors, not by this analysis project.

Related benchmark: Pei and colleagues, Neural Latents Benchmark '21, https://neurallatents.github.io/ . This supervised velocity study is not the official co-smoothing task and has no leaderboard claim. The official challenge stopped new submissions in January 2026 and released former evaluation targets; this project did not use those answer files or the official test NWB.

Anatomy mapping and task description: https://neurallatents.github.io/datasets . The source documentation identifies M1 by unit IDs beginning with 2 and warns of an electrode-index conversion error. Exact included unit IDs and conversion metadata are preserved. There are 90 M1 units in this study, not 96 electrode aggregates.

This analysis makes no clinical-validation or original-data-collection claim. The dataset authors do not endorse the analysis.
