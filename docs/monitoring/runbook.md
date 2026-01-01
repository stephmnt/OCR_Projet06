# Drift Runbook (MLOps)

## A. Data quality (prioritaire)
- verifier categories inconnues (CODE_GENDER, FLAG_OWN_CAR)
- verifier hausse des NaN / champs manquants
- verifier out-of-range numeriques
- verifier le taux de sentinelle DAYS_EMPLOYED
- verifier un changement de pipeline (mapping, imputation, schema)

## B. Prediction drift
- verifier la distribution des scores
- verifier le taux de classe positive
- verifier si le seuil metier a change

## C. Performance (si labels)
- AUC / logloss / Brier
- calibration (Platt/Isotonic)
- analyse par segment (region, canal, produit si dispo)

## Actions
- drift artificiel / bug data: corriger mapping ou schema, redeployer
- prior drift: recalibrer ou ajuster le seuil avec validation metier
- concept drift: retrain recent + validation temporelle + champion/challenger + plan de rollback

## Triggers
- Warning: drift data sans drift score ou perf
- Critical: drift data + drift score (et/ou perf en baisse)
- Retrain: drift persistant sur plusieurs fenetres + impact score/perf
