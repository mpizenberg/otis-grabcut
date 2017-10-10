# otis-grabcut

> [1] Outlining Objects for Interactive Segmentation on Touch Devices. 2017.
> Matthieu Pizenberg, Axel Carlier, Emmanuel Faure, Vincent Charvillat.
> In Proceedings of the 25th ACM International Conference on multimedia (MM'17).
> DOI: https://doi.org/10.1145/3123266.3123409

This repository contains a slightly modified version of GrabCut to take both
a background and a foreground initialization.
It is part of the [otis work][otis] so please cite the aforementioned paper in your work.

[otis]: https://github.com/mpizenberg/otis

This code is written in Matlab and based on
[Shai Bagon's graph cut wrapper for Matlab (GCMex)][shai-bagon].
It is a rewrite from the [work of Itay Blumenthal][grabcut-itay].

[shai-bagon]: https://github.com/shaibagon/GCMex
[grabcut-itay]: https://grabcut.weebly.com/

Please be aware that GCMex software can only be used for research purposes.
You should cite all the following papers in any resulting publication.

> [2] Efficient Approximate Energy Minimization via Graph Cuts
> Yuri Boykov, Olga Veksler, Ramin Zabih,
> IEEE transactions on PAMI, vol. 20, no. 12, p. 1222-1239, November 2001.
>
> [3] What Energy Functions can be Minimized via Graph Cuts?
> Vladimir Kolmogorov and Ramin Zabih.
> IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
> vol. 26, no. 2, February 2004, pp. 147-159.
>
> [4] An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.
> Yuri Boykov and Vladimir Kolmogorov.
> In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
> vol. 26, no. 9, September 2004, pp. 1124-1137.
>
> [5] Matlab Wrapper for Graph Cut. Shai Bagon.
> in www.wisdom.weizmann.ac.il/~bagon, December 2006.

## License

This sowftware is licensed under the GNU GPL.
