
COVID-19 impact on credit loss modelling
===

In this project, we aimed to improve the probability of default (PD) predictions of the Merton-Vasicek (MV) model. In normal times, the single-factor MV model can reasonably predict PD using a single economic driver for a large portfolio. However, in periods of complicated economic conditions such as the COVID-19 pandemic, where both the government restrictions and support measures affect the default rates, the single-factor MV model becomes unreliable. We addressed this issue by using a multi-factor MV model that incorporate multiple economic indicators and calibrating at sector level. The data used for this project was provided by [GCD](https://www.globalcreditdata.org/) and [SEB](https://www.seb.se). This package contains the codes for the multi-factor MV model. Further details and final results of the project can be found [here](https://sal.aalto.fi/files/teaching/ms-e2177/2021/2021-FinalReport-SEB-final.pdf).

Installation
---
We recommend using Python>=3.7 with a virtual environment. To install all dependencies installed, run:
```
pip install -r sebcreditrisk/requirements.txt
pip install -e sebcreditrisk
```

Running the models
---
Fitting the MV model with historical transition matrices `transitions_train` and corresponding economic indicators `z_train`, and predicting with `transitions_test`,  `z_test`:
```
import sebcreditrisk as scr

rating_level_rho = False	# Using same rho factor for all ratings
MV = scr.fit_MV(transitions_train, z_train, rating_level_rho=rating_level_rho)
pd_pred, pd_true = scr.predict_pd(MV, transitions_train, z_train, transitions_test, z_test)
```

