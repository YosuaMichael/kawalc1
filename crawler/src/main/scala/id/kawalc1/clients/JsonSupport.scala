package id.kawalc1.clients

import java.sql.Timestamp
import java.time.Instant

import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpjson4s.Json4sSupport
import enumeratum.values._
import id.kawalc1._
import org.json4s.JsonAST.{JInt, JObject}
import org.json4s.native.Serialization
import org.json4s.{CustomSerializer, Formats, native}


trait JsonSupport extends Json4sSupport with LazyLogging {
  implicit val serialization: Serialization.type = native.Serialization

  private def parseSummary(c1: Option[C1], summary: JObject): Option[Summary] = {
    c1 match {
      case Some(C1(Plano.YES, FormType.PPWP)) => Some(summary.extract[PresidentialLembar2])
      case Some(C1(Plano.YES, FormType.DPR)) => Some(summary.extract[Dpr])
      case None => Some(summary.extract[Pending])
      case _ => None
    }
  }

  case object SummarySerialize extends CustomSerializer[Verification](format => ( {
    case JObject(List(("c1", c1Json: JObject), ("sum", sum: JObject), ("ts", JInt(tss)))) =>
      val timeStamp = Timestamp.from(Instant.ofEpochMilli(tss.toLong))
      val maybeC1 = c1Json.extract[Option[C1]]
      val summary = sum match {
        case JObject(List(("jum", JInt(num)))) => Some(SingleSum(num.toInt))
        case JObject(List(("pJum", JInt(num)))) => Some(LegislativeSum(num.toInt))
        case _: JObject => parseSummary(maybeC1, sum)
      }
      Verification(timeStamp, maybeC1, summary)
  }, {
    case x: Verification => throw new Exception("cannot serialize")
  }
  )
  )

  import org.json4s.DefaultFormats

  val standardFormats: Formats = DefaultFormats + SummarySerialize ++ Seq(Json4s.serializer(FormType), Json4s.serializer(Plano))

  implicit val formats: Formats = standardFormats
}