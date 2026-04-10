# 🌂 Projeto Guarda-Chuva

Repositório oficial do projeto:  
🔗 https://github.com/Projeto-Guarda-Chuva

---

## 📌 Regras de Utilização do GitHub

Este documento define as boas práticas para organização e colaboração no repositório.

---

## 🧾 Commits

### ✅ Diretrizes Gerais
- Todo commit deve conter:
  - Código
  - Mensagem descritiva do que foi feito
- Priorizar:
  - Commits **frequentes**
  - Commits **pequenos**

---

### 🏷️ Convenções para Mensagens de Commit

As mensagens devem ser **claras, objetivas e descritivas**, indicando exatamente o que foi modificado.

#### 📌 Formato Padrão:
- [prefixo [número do grupo]] : mensagem

#### 📌 Prefixos:
- `feat` → Nova funcionalidade
- `fix` → Correção de bugs
- `docs` → Alterações na documentação
- `rem` → Remoção de código

#### 📌 Exemplos:
- git commit -m "feat [6.2] : Criação da função 'audio_captado'"
- git commit -m "fix [4.1] : Corrige bug na função 'audio_captado'"
- git commit -m "docs [3.2] : Alteração no README"
- git commit -m "rem [5.1] : Remoção da função 'audio_captado'"

---

### 🌿 Branching
#### 📌 Estrutura de Branches
- main
  - Contém versões estáveis do projeto
- Branches de desenvolvimento:
  - Criadas para cada funcionalidade
  - Criadas para correção de bugs
 
#### 🏷️ Padrão de Nomenclatura
- tipo/nome-da-branch
#### 📌 Exemplos:
- feat/nome-da-funcionalidade
- fix/nome-do-erro
- docs/atualizacao-readme

#### 🔄 Regras de Uso:
- Cada funcionalidade deve ter sua própria branch
- Correções de bugs devem ser feitas em branches específicas
- Uma branch só deve ser mesclada na main quando:
  - O desenvolvimento estiver finalizado

---
 
### 🔀 Pull Requests
#### 📌 Regras
- Todo código deve passar por revisão antes de ser integrado
- Um Pull Request deve:
  - Ser revisado por no mínimo 2 membros
  - Ser aprovado antes do merge
#### 👥 Revisão de Código
- A revisão deve ser feita por:
  - Membros da equipe
  - E/ou equipe de segurança
- O revisor deve:
  - Solicitar mudanças, se necessário
  - Garantir qualidade e boas práticas
#### ✅ Boas Práticas Gerais
- Escreva código limpo e organizado
- Documente sempre que necessário
- Siga os padrões definidos neste documento
- Mantenha comunicação clara com a equipe
