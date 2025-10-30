@echo off
REM ====================================================================
REM 10 Test Queries for VaultGemma DP Model
REM Model: models\vaultgemma_dp_20251017_062150
REM ====================================================================

echo.
echo ====================================================================
echo   VaultGemma DP Model - 10 Test Queries
echo ====================================================================
echo   Model: models\vaultgemma_dp_20251017_062150
echo   Training Stats: F1=0.231, EM=0.000
echo ====================================================================
echo.

echo.
echo ====================================================================
echo Query 1: Medication for Sara Moon
echo Expected: Amlodipine 5mg
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Sara Moon"
echo.
pause

echo.
echo ====================================================================
echo Query 2: Diagnosis for Donald Miller
echo Expected: Hyperlipidemia
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for Donald Miller"
echo.
pause

echo.
echo ====================================================================
echo Query 3: Medication for Lindsey Thompson
echo Expected: Metformin 500mg
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What medication is Lindsey Thompson taking"
echo.
pause

echo.
echo ====================================================================
echo Query 4: Diagnosis for Robert Smith
echo Expected: Chronic Obstructive Pulmonary Disease
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What condition was Robert Smith diagnosed with"
echo.
pause

echo.
echo ====================================================================
echo Query 5: Medication for Mary Davis
echo Expected: Gabapentin 300mg
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "Medication for Mary Davis"
echo.
pause

echo.
echo ====================================================================
echo Query 6: Diagnosis for Nancy Brown
echo Expected: Gastroesophageal Reflux Disease
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "Diagnosis for Nancy Brown"
echo.
pause

echo.
echo ====================================================================
echo Query 7: Medication for Kristen Warner
echo Expected: Omeprazole 20mg
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Kristen Warner"
echo.
pause

echo.
echo ====================================================================
echo Query 8: Diagnosis for Mr. Richard Smith
echo Expected: Depression
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What's Mr. Richard Smith's medical condition"
echo.
pause

echo.
echo ====================================================================
echo Query 9: Medication for Jennifer Davis
echo Expected: Amlodipine 5mg
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "Which drug was prescribed to Jennifer Davis"
echo.
pause

echo.
echo ====================================================================
echo Query 10: Diagnosis for David Frye
echo Expected: Anxiety Disorder
echo ====================================================================
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for David Frye"
echo.

echo.
echo ====================================================================
echo   All 10 Tests Completed!
echo ====================================================================
echo.
pause

