"""
Amar Passport — Bangladesh E-Passport Readiness AI Agent
Educational project. Always verify eligibility, documents, and fees on official portals.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "passport_reference.json"
OFFICIAL_FEE_URL = "https://www.epassport.gov.bd/instructions/passport-fees"

load_dotenv(ROOT / ".env")


def load_reference_data() -> dict[str, Any]:
    """Load local fallback reference data; fail clearly if the file is missing."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Local fallback database not found: {DATA_FILE}")
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_fetch_official_page(timeout: int = 8) -> dict[str, Any]:
    """
    Best-effort scrape of the official fee page.
    This is deliberately not used to overwrite local fees automatically:
    page markup can change, so the local reviewed fee table remains the fallback.
    """
    try:
        response = requests.get(
            OFFICIAL_FEE_URL,
            timeout=timeout,
            headers={"User-Agent": "AmarPassportEducationalProject/1.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.stripped_strings)
        if not text or "e-Passport" not in text:
            raise ValueError("Official page returned unexpected content.")
        return {
            "status": "success",
            "source": OFFICIAL_FEE_URL,
            "note": "Official page fetched. Fees are still calculated from the reviewed local reference table.",
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "source": str(DATA_FILE),
            "note": f"Could not fetch official page ({type(exc).__name__}: {exc}). Using local fallback data.",
        }


def make_llm() -> LLM:
    """Configure CrewAI to use the supplied OpenAI-compatible endpoint."""
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.cheaperinference.com/v1")
    model = os.getenv("MODEL_NAME", "glm-5.3")
    if not api_key or api_key == "put_your_api_key_here":
        raise EnvironmentError(
            "API key is missing. Copy .env.example to .env and set API_KEY. "
            "Never commit .env or share your API key."
        )
    return LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )


def validate_profile(profile: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Deterministic policy guardrails before the agents produce their report."""
    errors = []
    warnings = []
    age = profile.get("age")
    if not isinstance(age, int) or age < 0 or age > 120:
        errors.append("Age must be an integer between 0 and 120.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    requested_validity = profile.get("requested_validity")
    requested_pages = profile.get("pages", 48)
    delivery = profile.get("delivery", "regular").lower().replace(" ", "_")
    if requested_validity not in (None, 5, 10):
        errors.append("Requested validity must be 5, 10, or omitted.")
    if requested_pages not in (48, 64):
        errors.append("Pages must be 48 or 64.")
    if delivery not in ("regular", "express", "super_express"):
        errors.append("Delivery must be regular, express, or super_express.")

    if age < 18:
        allowed_validity = 5
        allowed_pages = [48]
        id_guidance = "English-version Birth Registration Certificate (BRC); confirm exact requirements with the passport office."
        if requested_validity == 10:
            errors.append("Eligibility conflict: applicants under 18 are generally eligible for a 5-year, 48-page passport, not 10 years.")
        if requested_pages == 64:
            errors.append("Eligibility conflict: applicants under 18 are generally issued a 48-page passport.")
    elif age > 65:
        # The assignment mentions a restriction for over-65 applicants. Avoid inventing a definitive
        # policy where official rules may vary; flag for manual confirmation.
        allowed_validity = "Confirm with passport office"
        allowed_pages = [48, 64]
        id_guidance = "NID or Birth Registration may be applicable depending on records; verify with the passport office."
        warnings.append("Applicant is over 65. Confirm validity and identity-document requirements with the passport office; this project does not assume a definitive restriction.")
    else:
        allowed_validity = "5 or 10 years, subject to official eligibility"
        allowed_pages = [48, 64]
        id_guidance = "NID for eligible adults; use BRC where permitted and confirm record matching requirements."
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "age": age,
        "allowed_validity": allowed_validity,
        "allowed_pages": allowed_pages,
        "id_guidance": id_guidance,
        "delivery": delivery,
    }


def calculate_fee(profile: dict[str, Any], reference: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Calculate fee from local table (already inclusive of 15% VAT)."""
    if not policy["valid"]:
        return {"fee_bdt": None, "note": "Fee not finalized because eligibility/profile validation failed."}
    age = profile["age"]
    validity = profile.get("requested_validity")
    if age < 18:
        validity = 5
    elif validity not in (5, 10):
        validity = 10 if age <= 65 else 5
    pages = profile.get("pages", 48)
    if age < 18:
        pages = 48
    delivery = policy["delivery"]
    fee = reference["fees_2026"][f"{pages}_pages"][f"{validity}_years"][delivery]
    return {
        "pages": pages,
        "validity_years": validity,
        "delivery": delivery,
        "fee_bdt": fee,
        "vat_included": True,
        "note": "Published domestic fee table already includes 15% VAT; do not add VAT again.",
    }


def build_agents(llm: LLM, reference: dict[str, Any]) -> tuple[Agent, Agent, Agent]:
    policy_agent = Agent(
        role="Bangladesh Passport Policy Guardian",
        goal="Assess age-related passport validity and identity-document guidance, flag contradictions, and distinguish verified rules from items needing official confirmation.",
        backstory=(
            "You are a careful Bangladesh e-passport policy specialist. You prioritize official sources, "
            "never invent legal rules, and clearly flag uncertainty, especially for minors and applicants over 65."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    fee_agent = Agent(
        role="Chancellor of the Exchequer — Passport Fee Auditor",
        goal="Calculate the exact domestic passport fee from the supplied 2026 local table, which already includes 15% VAT.",
        backstory=(
            "You are a meticulous financial auditor. Never estimate a fee when a table value is available. "
            "Never add VAT twice. Use only the provided reference data."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    docs_agent = Agent(
        role="Document Architect — Checklist Specialist",
        goal="Create a tailored passport document checklist based on age, profession, name-change circumstances, and application type.",
        backstory=(
            "You are a documentation officer. You customize checklists for minors, government employees, "
            "and name changes while marking items that require confirmation."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return policy_agent, fee_agent, docs_agent


def create_report(profile: dict[str, Any]) -> str:
    reference = load_reference_data()
    fetch_status = try_fetch_official_page()
    policy = validate_profile(profile, reference)
    fee = calculate_fee(profile, reference, policy)

    if not policy["valid"]:
        return render_validation_report(profile, policy, fetch_status)

    try:
        llm = make_llm()
        policy_agent, fee_agent, docs_agent = build_agents(llm, reference)

        policy_task = Task(
            description=(
                "Review this applicant profile and deterministic guardrails. Do not override hard validation errors. "
                "Explain eligibility, validity, identity-document guidance, and uncertainties. "
                f"Profile: {json.dumps(profile, ensure_ascii=False)}\n"
                f"Guardrails: {json.dumps(policy, ensure_ascii=False)}\n"
                "Reference data: " + json.dumps(reference, ensure_ascii=False)
            ),
            expected_output="A concise eligibility assessment with any restrictions or flags.",
            agent=policy_agent,
        )
        fee_task = Task(
            description=(
                "Use the Policy Guardian output as context and the supplied deterministic fee result. "
                "Report pages, validity, delivery, fee in BDT, and state VAT is included. Do not change the calculated fee. "
                f"Calculated fee record: {json.dumps(fee, ensure_ascii=False)}\n"
                f"Reference fee table: {json.dumps(reference['fees_2026'], ensure_ascii=False)}"
            ),
            expected_output="A fee audit summary including the exact BDT amount and VAT note.",
            agent=fee_agent,
            context=[policy_task],
        )
        docs_task = Task(
            description=(
                "Prepare a customized checklist using the applicant profile and local reference documents. "
                "Include government NOC/GO if relevant, parent NIDs for minors, and marriage/name-change proof only if applicable. "
                "Mark items as 'verify if applicable' where facts are missing. Provide both English and Bangla. "
                f"Profile: {json.dumps(profile, ensure_ascii=False)}\n"
                f"Document reference: {json.dumps(reference['required_docs'], ensure_ascii=False)}"
            ),
            expected_output="An actionable English and Bangla document checklist.",
            agent=docs_agent,
        )
        crew = Crew(
            agents=[policy_agent, fee_agent, docs_agent],
            tasks=[policy_task, fee_task, docs_task],
            process=Process.sequential,
            verbose=True,
        )
        agent_output = str(crew.kickoff())
        return render_final_report(profile, policy, fee, fetch_status, agent_output)
    except Exception as exc:
        # Fallback report remains useful even when API credentials, network, or LLM calls fail.
        return render_final_report(
            profile, policy, fee, fetch_status,
            f"AI-agent fallback activated: {type(exc).__name__}: {exc}\n"
            "The deterministic local policy checks, fee calculator, and checklist are used below."
        )


def render_validation_report(profile: dict[str, Any], policy: dict[str, Any], fetch_status: dict[str, Any]) -> str:
    rows = [
        ("Applicant age", str(profile.get("age"))),
        ("Eligibility status", "FLAGGED — manual correction required"),
        ("Issues", "; ".join(policy["errors"])),
        ("Data source", fetch_status["source"]),
    ]
    return "# Amar Passport — Readiness Report\n\n" + markdown_table(rows) + (
        "\n\n## Recommended action\nCorrect the profile or confirm the exception with the Department of Immigration and Passports before paying.\n"
        "\n## বাংলা সারাংশ\nআবেদনের তথ্যের মধ্যে অসামঞ্জস্য পাওয়া গেছে। ফি চূড়ান্ত করার আগে বয়স, পাসপোর্টের মেয়াদ ও পৃষ্ঠাসংখ্যা যাচাই করুন।\n"
    )


def markdown_table(rows: list[tuple[str, str]]) -> str:
    escaped_rows = []
    for key, value in rows:
        safe_value = str(value).replace("|", "\\|").replace("\n", " ")
        escaped_rows.append(f"| {key} | {safe_value} |")
    return "| Field | Result |\n|---|---|\n" + "\n".join(escaped_rows)


def render_final_report(profile: dict[str, Any], policy: dict[str, Any], fee: dict[str, Any],
                        fetch_status: dict[str, Any], agent_output: str) -> str:
    reference = load_reference_data()
    profession = str(profile.get("profession", "not specified")).lower()
    docs = list(reference["required_docs"]["adult"])
    # Keep commonly requested application documents explicit; verify exact requirements officially.
    if profile.get("age", 18) >= 18:
        docs.extend(["Profession proof / employment certificate (if applicable)", "Application Summary"])
    if profile.get("age", 18) < 18:
        docs = list(reference["required_docs"]["minor_under_18"])
    if "government" in profession or "govt" in profession:
        docs.extend(reference["required_docs"]["government_staff"])
    if profile.get("name_change"):
        docs.append("Marriage certificate / court order or other supporting name-change evidence, as applicable")
    if profile.get("previous_passport"):
        docs.append("Previous passport")
    docs = list(dict.fromkeys(docs))

    rows = [
        ("Age / বয়স", str(profile.get("age"))),
        ("Profession / পেশা", str(profile.get("profession", "Not provided"))),
        ("Application location / আবেদনের স্থান", str(profile.get("location", "Not provided"))),
        ("Urgency / ভ্রমণের সময়সীমা", str(profile.get("urgency", "Not specified"))),
        ("Eligibility / যোগ্যতা", "Eligible subject to official verification" if policy["valid"] else "FLAGGED"),
        ("Passport validity / মেয়াদ", f"{fee.get('validity_years', 'Needs confirmation')} years" if fee.get("fee_bdt") else str(policy["allowed_validity"])),
        ("Pages / পৃষ্ঠা", str(fee.get("pages", profile.get("pages", 48)))),
        ("Delivery / ডেলিভারি", str(fee.get("delivery", policy["delivery"]))),
        ("Total fee / মোট ফি", f"BDT {fee['fee_bdt']:,}" if fee.get("fee_bdt") else "Not calculated"),
        ("VAT / ভ্যাট", "Included (15%) / অন্তর্ভুক্ত"),
        ("Identity guidance / পরিচয়পত্র", policy["id_guidance"]),
        ("Data retrieval / তথ্য সংগ্রহ", fetch_status["status"] + " — " + fetch_status["note"]),
    ]
    checklist = "\n".join(f"- [ ] {item}" for item in docs)
    bangla_checklist = "\n".join(f"- [ ] {item}" for item in docs)
    warnings = policy.get("warnings", [])
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- No additional automated warning."
    return f"""# Amar Passport — Passport Readiness Report

## 1. Applicant summary (English)

{markdown_table(rows)}

## 2. Customized document checklist (English)

{checklist}

## 3. বাংলা সারসংক্ষেপ

| বিষয় | ফলাফল |
|---|---|
| বয়স | {profile.get('age')} বছর |
| পেশা | {profile.get('profession', 'উল্লেখ করা হয়নি')} |
| পাসপোর্টের মেয়াদ | {fee.get('validity_years', 'যাচাই প্রয়োজন')} বছর |
| পৃষ্ঠাসংখ্যা | {fee.get('pages', profile.get('pages', 48))} |
| ডেলিভারি | {fee.get('delivery', policy['delivery'])} |
| মোট ফি | {'৳' + format(fee['fee_bdt'], ',') if fee.get('fee_bdt') else 'হিসাব করা হয়নি'} |
| ভ্যাট | ১৫% ভ্যাট অন্তর্ভুক্ত |

### প্রয়োজনীয় কাগজপত্রের তালিকা

{bangla_checklist}

## 4. Policy flags and limitations

{warning_text}

- Fees shown are for applications made inside Bangladesh and are based on the local reference table.
- Delivery times are estimates, not guarantees.
- Identity documents and special-case requirements must be checked against the official portal and the relevant passport office.
- Applicants over 65 are flagged for manual confirmation because this educational project does not encode a definitive age-specific exception.

## 5. AI agent execution / fallback note

{agent_output}

## Official source

{OFFICIAL_FEE_URL}

*Educational aid only; this is not an official government decision or legal advice.*
"""


def _ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    """Ask a yes/no question, allowing a default when one is supplied."""
    suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
    while True:
        answer = input(prompt + suffix + ": ").strip().lower()
        if not answer and default is not None:
            return default
        if answer in {"y", "yes", "হ্যাঁ"}:
            return True
        if answer in {"n", "no", "না"}:
            return False
        print("Please enter yes or no (y/n).")


def _extract_age(sentence: str) -> int | None:
    patterns = [r"\b(\d{1,3})\s*[- ]?year[- ]?old\b", r"\bage\s*(?:is|:)?\s*(\d{1,3})\b"]
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def ask_profile() -> dict[str, Any]:
    """Collect a natural-language profile, then ask only for missing details."""
    print("=== Amar Passport: Bangladesh E-Passport Readiness ===")
    print("Describe your passport needs in one sentence. Example:")
    print('I am a 24-year-old private employee. I need a 64-page, 10-year passport as express. I have an NID and live in Dhaka.')
    sentence = input("Your description: ").strip()

    age = _extract_age(sentence)
    while age is None or age < 1 or age > 120:
        if age is not None:
            print(f"The age '{age}' looks invalid. Please confirm an age from 1 to 120 (for example, 15 or 21).")
        raw = input("Applicant age (number): ").strip()
        try:
            age = int(raw)
        except ValueError:
            age = None

    lower = sentence.lower()
    profession = ""
    if re.search(r"\b(student|stusent|studnt|pupil|school student|university student)\b", lower):
        profession = "student"
    elif re.search(r"\b(private[- ]?(sector )?employee|private[- ]?job|private employee|corporate employee)\b", lower):
        profession = "private sector employee"
    elif re.search(r"\b(government employee|govt employee|government staff|civil servant)\b", lower):
        profession = "government employee"
    elif re.search(r"\b(self[- ]?employed|business owner|businessman|businesswoman|entrepreneur)\b", lower):
        profession = "self-employed / business"
    if not profession:
        profession = input("Profession (student/private employee/government employee/etc.): ").strip()

    page_match = re.search(r"\b(48|64)\s*[- ]?(?:page|pages)\b|\b(?:pages?\s*)(48|64)\b", lower)
    pages = int(next(group for group in page_match.groups() if group)) if page_match else None
    while pages not in (48, 64):
        raw = input("Passport pages [48/64] (default 48): ").strip() or "48"
        try:
            pages = int(raw)
        except ValueError:
            pages = None
        if pages not in (48, 64):
            print("Please enter 48 or 64.")

    validity_match = re.search(r"\b(5|10)\s*[- ]?year(?:s)?\b|\bvalidity\s*(?:of|:)?\s*(5|10)\b", lower)
    validity = int(next(group for group in validity_match.groups() if group)) if validity_match else None
    if validity is None:
        recommended = 5 if age < 18 else 10
        if age < 18:
            print("For an applicant under 18, this project applies the 5-year validity guardrail.")
            validity = 5
        else:
            raw = input("Requested validity [5/10] (default 10 years for adults): ").strip() or str(recommended)
            while raw not in {"5", "10"}:
                raw = input("Please enter 5 or 10: ").strip()
            validity = int(raw)

    if re.search(r"\bsuper[- _]?express\b", lower):
        delivery = "super_express"
    elif re.search(r"\bexpress\b|\burgent\b|\bwithin\s+(?:one|two|1|2)\s+weeks?\b", lower):
        delivery = "express"
    elif re.search(r"\bregular\b|\bnormal delivery\b", lower):
        delivery = "regular"
    else:
        raw = input("Delivery [regular/express/super_express] (default regular): ").strip().lower() or "regular"
        aliases = {"super express": "super_express", "super-express": "super_express", "normal": "regular"}
        delivery = aliases.get(raw, raw)
        while delivery not in {"regular", "express", "super_express"}:
            raw = input("Please enter regular, express, or super_express: ").strip().lower()
            delivery = aliases.get(raw, raw)

    nid_match = re.search(r"\b(no|don't have|do not have|without)\s+(?:an?\s+)?nid\b", lower)
    has_nid = "no" if nid_match else ("yes" if re.search(r"\b(?:have|with)\s+(?:an?\s+)?nid\b|\bnid\b", lower) else "unknown")
    if has_nid == "unknown":
        has_nid = "yes" if _ask_yes_no("Do you have a National ID (NID)?") else "no"

    brc_match = re.search(r"\b(?:birth registration|brc)\b", lower)
    if brc_match:
        has_brc = "no" if re.search(r"\b(?:no|without|don't have|do not have)\s+(?:an?\s+)?(?:english\s+)?(?:birth registration|brc)", lower) else "yes"
    elif age < 18:
        has_brc = "yes" if _ask_yes_no("Do you have an English Birth Registration Certificate (BRC)?") else "no"
    else:
        has_brc = "unknown"

    location_match = re.search(r"\b(?:live|living|reside|residing)\s+in\s+([A-Za-z][A-Za-z .'-]*?)(?:[.,;]|$)", sentence, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else ""
    if not location:
        location = input("City/district where you will apply (e.g., Dhaka): ").strip()

    name_change = bool(re.search(r"\b(name changed|name change|changed my name|different name)\b", lower))
    if not name_change:
        name_change = _ask_yes_no("Have you changed your name or do your documents show different names?", default=False)
    previous_passport = bool(re.search(r"\b(previous passport|old passport|renewal)\b", lower)) and not bool(re.search(r"\b(no|never|don't have|do not have)\s+(?:a\s+)?(?:previous|old) passport\b", lower))
    if not re.search(r"\b(previous passport|old passport|renewal)\b", lower):
        previous_passport = _ask_yes_no("Do you have a previous passport?", default=False)

    urgency_match = re.search(r"\b(?:urgent|business trip|travel|flight|trip)\b[^.!?;]*", sentence, re.IGNORECASE)
    urgency = urgency_match.group(0).strip() if urgency_match else "Not specified"
    if urgency == "Not specified":
        urgency = input("Any travel deadline or urgency? (press Enter if none): ").strip() or "Not specified"

    if age < 18 and (validity == 10 or pages == 64):
        print("\nEligibility warning: this project flags applicants under 18 for a 5-year, 48-page passport.")
        if _ask_yes_no("Would you like to use 5-year validity and 48 pages for this profile?", default=True):
            validity, pages = 5, 48
        else:
            raise ValueError("Profile conflicts with the under-18 guardrail. Correct the details before generating a report.")

    profile = {
        "age": age,
        "profession": profession,
        "pages": pages,
        "requested_validity": validity,
        "delivery": delivery,
        "urgency": urgency,
        "location": location,
        "has_nid": has_nid,
        "has_birth_registration": has_brc,
        "name_change": name_change,
        "previous_passport": previous_passport,
    }

    print("\nExtracted profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    if not _ask_yes_no("Is this information correct?", default=True):
        print("Please run the program again and correct your description when prompted.")
        raise ValueError("Applicant profile was not confirmed.")
    return profile


def main() -> None:
    try:
        profile = ask_profile()
        report = create_report(profile)
        output_file = ROOT / "passport_readiness_report.md"
        output_file.write_text(report, encoding="utf-8")
        print("\n" + report)
        print(f"\nReport saved to: {output_file}")
    except (ValueError, FileNotFoundError, EnvironmentError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
