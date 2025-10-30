@echo off
REM ====================================================================
REM 5 Test Queries for VaultGemma DP Model
REM Model: models\vaultgemma_dp_20251017_062150
REM ====================================================================

echo.
echo ====================================================================
echo   VaultGemma DP Model - 5 Test Queries
echo ====================================================================
echo   Model: models\vaultgemma_dp_20251017_062150
echo   Training Stats: F1=0.231, EM=0.000
echo ====================================================================
echo.

echo.
echo ====================================================================
echo TEST 1: Diagnosis for Karen Hooper
echo Expected: Sleep Apnea
echo ====================================================================
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What is the diagnosis for Karen Hooper?" --gen-beams 1 --gen-max-new-tokens 6
echo.
pause

echo.
echo ====================================================================
echo TEST 2: Medication for Sara Moon
echo Expected: Amlodipine 5mg
echo ====================================================================
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What medication is Sara Moon taking?" --gen-beams 1 --gen-max-new-tokens 6
echo.
pause

echo.
echo ====================================================================
echo TEST 3: Condition for Lindsey Thompson
echo Expected: Type 2 Diabetes
echo ====================================================================
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What condition was Lindsey Thompson diagnosed with?" --gen-beams 1 --gen-max-new-tokens 8
echo.
pause

echo.
echo ====================================================================
echo TEST 4: Medication for Donald Miller
echo Expected: Lorazepam 0.5mg
echo ====================================================================
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "Medication for Donald Miller?" --gen-beams 1 --gen-max-new-tokens 6
echo.
pause

echo.
echo ====================================================================
echo TEST 5: Diagnosis for Kathleen Howard
echo Expected: Asthma
echo ====================================================================
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "Diagnosis for Kathleen Howard?" --gen-beams 1 --gen-max-new-tokens 4
echo.

echo.
echo ====================================================================
echo   All 5 Tests Completed!
echo ====================================================================
echo.
pause

