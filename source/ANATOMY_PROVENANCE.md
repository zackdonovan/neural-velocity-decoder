# Region and source interpretation

Official NLB dataset documentation: https://neurallatents.github.io/datasets (MC_Maze section, verified November 6, 2026).

It states: “for units recorded from M1, the leading digit is 2.” It also explains a conversion error in M1 electrode indices. This analysis follows the documented unit-ID rule, not the erroneous electrode index. Unit identities will be preserved exactly; they are not equivalent to the 96 aggregated M1 electrode features in the Indy/Loco studies.

The task contains delayed, event-aligned straight and curved maze reaches, unlike continuous random-target reaching. This is a custom supervised velocity decoder, not the NLB co-smoothing competition. The official training-trial subset can interleave excluded validation/test trials; chronology refers to the available training-tagged trials, not uninterrupted session segments.

Direct official asset metadata identifies subject Jenkins and recording date September 25, 2009. We have not collected the data or established new clinical capability. Official offline velocity processing can use position information unavailable at the current bin endpoint. Counts ending at that endpoint decode concurrent movement only after the bin completes, not future movement.
